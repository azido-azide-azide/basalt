#include "common.hpp"
#include "record.hpp"
#include "expert_plan.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/models/qwen4exp/kernels/cvec.hpp"
#include "basalt/models/qwen4exp/kernels/tp_exchange.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/core/tp_mailbox.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"
#include <string>

namespace basalt::core {
using namespace verify_detail;

// The decode window's overlap structure (all always on): in windows of 2+ tokens the shared expert's side kernels run
// at the greatest stream priority, so their blocks are scheduled ahead of the routed experts' queued blocks; in windows
// of 1-2 tokens a GDN layer's z projection is forked at that priority beside the qkv projection; the MoE route is
// split (CUDA0's plan and q8_1 rows on the main stream, the ring for the peer card and the CPU pool on a stream of its
// own) with the hc read's router items writing the FFN input's q8_1 rows and the gate/up launch building CUDA0's plan
// itself; the combine runs as the wait kernel's programmatic dependent.  A QSA layer's K, V and the indexer key/query
// run on branches of their own beside the query projection; its block scores skip a window whose selections are all
// the identity; the head's hyper-connection read does every column in one launch; layer 1's PLE takes two launches;
// the attention's merge also writes the output projection's q8_1 rows.  The attention's hc read writes the
// projections' q8_1 rows, and a GDN layer's qkv and z projections are one dense_tc_pair launch.

// ================================ THE WINDOW, AS CAPTURED ================================
//
// The stream is ordered pre(0) | post(0) pre(1) | post(1) pre(2) | ...: each layer's mixer and router, then its
// experts and combine, with the next layer's mixer launched behind them.
bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {
    WindowRecorder recorder(*this, T, cs, err);
    return recorder.run();
}

// The pair's card turns its own heads' activation rows into both cards' whole rows: its values land at their
// model head slot (zeros elsewhere), the mailbox sums the two cards' placements (x + 0 = x: one rounding, equal
// bits on both cards), and the widened row quantizes as every single-card window quantizes its projection input.
//
// `gdn_heads` says the rows are a GDN layer's, which changes ONLY where a row lands.  A GDN card's value heads
// come from a key-head-group cut (core::gdn_head_split: value head v reads key head v % h_k), so its packed
// head p is model head (p / nk) * h_k + k0 + p % nk and belongs at THAT head's columns of the replicated
// output projection.  The plain `off = rank * half` is the model slot only when the card's heads are
// contiguous (h_v == h_k, one run); with 48 heads over 16 key heads it hands the projection heads in the wrong
// order - 32 of 48 of them - which is deterministic, bit-for-bit equal on both cards, and wrong from the first
// window's first layer: the pair's token salad.
bool Verifier::WindowRecorder::tp_assemble(int64_t half_dim, int64_t full_dim, const void* src, bool src_bf16,
                                              int site, bool gdn_heads) {
    using namespace basalt::kernels;
    FnTp* tp = v_.tp_;
    const int64_t off = (int64_t) tp->rank * half_dim;
    int64_t sg = 0, dg = 0, base = 0;
    if (gdn_heads) {
        const GdnHeadSlice& gs = tp->gdn;
        if (gs.nk <= 0 || gs.h_k <= 0 || gs.h_v <= 0 || gs.s <= 0 || gs.nv() * gs.s != half_dim ||
            (int64_t) gs.h_v * gs.s != full_dim || gs.h_v % gs.h_k != 0 || gs.nv() % gs.nk != 0 ||
            gs.h_v / gs.h_k != gs.nv() / gs.nk) {
            err = "tp mixer exchange [call site line " + std::to_string(site) +
                  "]: the card's GDN head slice does not describe these rows (nk=" + std::to_string(gs.nk) +
                  " h_k=" + std::to_string(gs.h_k) + " h_v=" + std::to_string(gs.h_v) +
                  " s=" + std::to_string(gs.s) + " half=" + std::to_string((long long) half_dim) +
                  " full=" + std::to_string((long long) full_dim) + ")";
            return false;
        }
        gdn_place_groups(gs, sg, dg, base);
    }
    if (src_bf16) {
        if (gdn_heads)
            tp_place_bf16_bf16_grouped(v_.mixb_, (const uint16_t*) src, half_dim, full_dim, sg, dg, base, T, cs);
        else
            tp_place_bf16_bf16(v_.mixb_, (const uint16_t*) src, half_dim, full_dim, off, T, cs);
    } else {
        if (gdn_heads)
            tp_place_bf16_f32_grouped(v_.mixb_, (const float*) src, half_dim, full_dim, sg, dg, base, T, cs);
        else
            tp_place_bf16_f32(v_.mixb_, (const float*) src, half_dim, full_dim, off, T, cs);
    }
    std::string e;
    if (!tp->mb->exchange_bf16(tp->rank, tp->epi[tp->rank]++ & 1, v_.mixb_, (size_t) T * full_dim * 2, cs, e)) {
        err = "tp mixer exchange [call site line " + std::to_string(site) + "] half=" + std::to_string((long long) half_dim) +
              " full=" + std::to_string((long long) full_dim) + " T=" + std::to_string(T) + " N=" +
              std::to_string((long long) N) + " card=" + std::to_string(tp->rank) + " epi=" +
              std::to_string((long long) tp->epi[tp->rank]) + " ZV=" + std::to_string((long long) g.ssm_value_dim) +
              " ZVF=" + std::to_string((long long) gf.ssm_value_dim) + " NH=" + std::to_string((long long) NH) +
              " HD=" + std::to_string((long long) HD) + " NKV=" + std::to_string((long long) NKV) + "]: " + e;
        return false;
    }
    tp_bf16_widen(v_.mixf_, v_.mixb_, (int64_t) T * full_dim, cs);
    native_quantize_q8_1(v_.mixf_, v_.xq_, (int) full_dim, T, cs);
    return true;
}

bool Verifier::WindowRecorder::run() {
    using namespace basalt::kernels;
    if (!inputs()) return false;
    fuse_head_gr = (v_.le_ == g.n_layers) && (v_.head_ != nullptr && v_.head_->loaded()) &&
                              !cvec().covers(g.n_layers - 1);
    if (!pre(v_.lb_)) return false;
    for (int64_t l = v_.lb_; l < v_.le_; ++l) {
        if (!post(l)) return false;
        if (v_.tp_diag_ != nullptr) {   // BASALT_TP_DIAG: residual + the pair-added MoE norm after each layer
            basalt::kernels::diag_norm(Rt(0), (int) N, v_.tp_diag_ + 8 * l, cs);
            basalt::kernels::diag_norm(v_.bo_, (int) N, v_.tp_diag_ + 8 * l + 1, cs);
        }
        if (l + 1 < v_.le_ && !pre(l + 1)) return false;
    }
    if (lz_open) cudaStreamWaitEvent(cs, v_.lz_join_, 0);
    if (v_.le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
        float* hout = v_.hand_out_ + (size_t) hrow0 * HB;
        copy_from_mapped(hout, v_.R_, (int64_t) T * HC * N, cs);
        copy_from_mapped(hout + (size_t) T * HC * N, v_.bo_, (int64_t) T * N, cs);
        copy_from_mapped(hout + (size_t) T * (HC + 1) * N, v_.inj2_, (int64_t) T * HC, cs);
        if (!v_.batch_rec_ && v_.next_ != nullptr && v_.next_->m_flag_ != nullptr) signal_flag(v_.next_->m_flag_ + 12, 1, cs);
        return true;
    }
    // the pair's rank 1 records no head: the LM head and its sampling are rank 0's alone (the twin's
    // output.weight entry carries no bytes on its card - tp_skip_head, verify.hpp) and the driver discards
    // rank 1's picks; the graph simply ends after the last layer's post().
    if (v_.tp_skip_head()) return true;
    return head();
}

}  // namespace basalt::core
