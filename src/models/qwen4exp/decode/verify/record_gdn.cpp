#include "common.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "record.hpp"
#include "expert_plan.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/kernels/gdn_window.hpp"
#include "basalt/models/qwen4exp/kernels/dense_tc.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"

namespace basalt::core {
using namespace verify_detail;

bool Verifier::WindowRecorder::gdn(int64_t l, const LayerView& v, bool read_q8, bool gs_split) {
    using namespace basalt::kernels;
    const int n = T;
    float* xm = v_.mixed_;
    // ======================= GDN =======================
    const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                    *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                    *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                    *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                    *wsa = need(v, "ssm_a", err);
    if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
    if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
        !native_of(wout, v.name("ssm_out.weight"), err))
        return false;
    const int64_t gi = gdn_idx[(size_t) l];
    float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
    float* qkv = v_.qkv_L_ + (size_t) gi * MT * C;
    float* hb = v_.h_L_ + (size_t) gi * MT * C;
    float* gate = v_.gate_L_ + (size_t) gi * MT * HV;
    float* beta = v_.beta_L_ + (size_t) gi * MT * HV;
    // The head-indexed tables (ssm_alpha/ssm_beta rows, ssm_dt.bias/ssm_a) are addressed by the launch's
    // head.  Under nk == 0 - a pair card running its own geometry, and solo's whole-layer path - that head is
    // the card's PACKED LOCAL head, and the card's local head p is model head (p / nk) * h_k + k0 + p % nk,
    // NOT rank * HV + p (the plan slices value heads by key-head group; core::gdn_head_split).  A pair card
    // therefore reads the rows Verifier::tp_gather_gdn_params laid out in its own order.  The gs / driven-peer
    // launch runs with nk > 0, where head_map already yields model heads, so it keeps the full tables.
    const FnGdnPar* par = nullptr;
    if (v_.tp_ != nullptr) {
        if (v_.tp_->gdn_par.size() <= (size_t) gi) {
            err = "GDN layer " + std::to_string(l) + ": the pair card has no gathered head rows (card " +
                  std::to_string(v_.tp_->rank) + " of the tensor pair)";
            return false;
        }
        par = &v_.tp_->gdn_par[(size_t) gi];
    }
    const uint16_t* w_alpha_p = par != nullptr ? par->alpha : (const uint16_t*) wa->data;
    const uint16_t* w_beta_p = par != nullptr ? par->beta : (const uint16_t*) wb->data;
    const float* dt_p = par != nullptr ? par->dt : (const float*) wdt->data;
    const float* ssm_a_p = par != nullptr ? par->a : (const float*) wsa->data;
    if (l == 0  && !v_.batch_rec_) {   // v9: what the card's table really holds for the attn weights
        basalt::log::raw("TPMETA rank=%d tp=%d l=0 wout_b=%llu wout_t=%d gate_b=%llu qkv_b=%llu conv_b=%llu "
                         "ZV=%lld ZVF=%lld C=%lld N=%lld HK=%lld HV=%lld alpha_b=%llu",
                         v_.tp_ ? v_.tp_->rank : -1, v_.tp_ ? 1 : 0,
                         (unsigned long long) wout->bytes, (int) wout->native_type,
                         (unsigned long long) wg->bytes, (unsigned long long) wqkv->bytes,
                         (unsigned long long) wc->bytes, (long long) ZV, (long long) ZVF, (long long) C,
                         (long long) N, (long long) HK, (long long) HV, (unsigned long long) wa->bytes);
    }
    PdlScope ps(pdl);   // the layer's GDN kernels, after the hc read
    if (gs_split) {   // CUDA0's key heads in place (dec_gdn_rec's head map); the peer's arrive in the combine
        const GsLayer& GL = v_.gs_layers_[(size_t) gi];
        const int C0 = (int) (C / HK * v_.gs_nk0_), Z0 = (int) (ZV / HK * v_.gs_nk0_);
        kernels::dense_tc_pair(GL.tq, GL.wqkv0, v_.gs_qkv0_, C0, GL.tz, GL.wz0, v_.gs_z0_, Z0, v_.xq_, (int) N, n, cs);
        DecRecArgs ra;
        ra.state = state; ra.qkv = v_.gs_qkv0_; ra.conv_w = (const float*) wc->data; ra.x = xm;
        ra.w_alpha = w_alpha_p; ra.w_beta = w_beta_p;
        ra.dt = dt_p; ra.ssm_a = ssm_a_p; ra.z = v_.gs_z0_;
        ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = v_.xq_;
        ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
        if (v_.tp_) ra.y16 = v_.y16_;   // the pair's card keeps its heads' bf16 rows for the exchange
        ra.k0 = 0; ra.nk = v_.gs_nk0_; ra.qkv_out = qkv;
        dec_gdn_rec(ra, n, cs);
        dense_proj(4, GL.to, GL.wo0, v_.xq_, v_.bo_, Z0, (int) N, n, cs);
        if (v_.tp_diag_bits_ != nullptr && l < 3) {
            basalt::kernels::diag_copy_i32((const int32_t*) v_.bo_, v_.tp_diag_bits_ + l * 288 + 120, 8, cs);
        }
        if (v_.tp_diag_h1_ != nullptr && l < 3) {   // v17: the gs serve's per-head xq hashes + its wo table + ptr
            basalt::kernels::diag_hash_u8((const uint8_t*) v_.xq_, 144, 24, v_.tp_diag_h1_ + l * 96, cs);
            if (v_.tp_diag_bits_ != nullptr) { const int32_t* xq=(const int32_t*) v_.xq_;
              basalt::kernels::diag_copy_i32(xq+0,   v_.tp_diag_bits_+l*288+128, 36, cs);
              basalt::kernels::diag_copy_i32(xq+288, v_.tp_diag_bits_+l*288+164, 36, cs);
              basalt::kernels::diag_copy_i32(xq+864, v_.tp_diag_bits_+l*288+200, 36, cs);
              basalt::kernels::diag_copy_i32(xq+1152,v_.tp_diag_bits_+l*288+236, 36, cs);
            }
            cudaMemsetAsync(v_.tp_diag_h1_ + l * 96 + 24, 0, 24 * sizeof(int32_t), cs);
            basalt::kernels::diag_hash_u8((const uint8_t*) GL.wo0, 64 * 2520, 40, v_.tp_diag_h1_ + l * 96 + 48, cs);
            int32_t pd[2] = {(int32_t) (uintptr_t) GL.wo0, (int32_t) ((uintptr_t) GL.wo0 >> 32)};
            cudaMemcpyAsync(v_.tp_diag_h1_ + l * 96 + 88, pd, 8, cudaMemcpyHostToDevice, cs);
        }
        if (v_.tp_diag_bits_ != nullptr && l < 3 ) {
            basalt::kernels::diag_copy_i32((const int32_t*) GL.wqkv0, v_.tp_diag_bits_ + l * 288 + 64, 8, cs);
            basalt::kernels::diag_copy_i32((const int32_t*) GL.wz0, v_.tp_diag_bits_ + l * 288 + 72, 8, cs);
            basalt::kernels::diag_copy_i32((const int32_t*) GL.wo0, v_.tp_diag_bits_ + l * 288 + 96, 8, cs);
            basalt::kernels::diag_copy_i32((const int32_t*) v_.xq_, v_.tp_diag_bits_ + l * 288 + 104, 8, cs);
        }
        goto gdn_done;
    }
    // conv, gates, recurrence, gated norm and the output's q8_1 in dec_gdn_rec's one or two launches; at
    // 2/3 the z projection runs on the side stream beside the recurrence (gdn_window.hpp)
    {
    // a batch window may take the fused recurrence where the qkv and z projections are one pair launch
    const bool pair_ok = kernels::dense_tc_pair_supported(wqkv->native_type, wg->native_type, (int) N) &&
                         ((uintptr_t) wqkv->native_data & 15) == 0 && ((uintptr_t) wg->native_data & 15) == 0;
    const int gdnf = ((!v_.batch_rec_ || pair_ok) && N == 2560 && n <= kVerifyMaxT) ? dec_gdnf_mode() : 0;
    auto rec = [&](const DecRecArgs& ra) {
        if (v_.batch_rec_) dec_gdn_rec_seg(ra, v_.segb_, (size_t) gi * gdn_floats, n, n, cs);
        else dec_gdn_rec(ra, n, cs);
    };
    const bool gdnf_fork = gdnf >= 2 && v_.sh_cs_ != nullptr && v_.ev_fork_ != nullptr && v_.ev_join_ != nullptr;
    auto gdnf_z_side = [&] {
        cudaEventRecord(v_.ev_fork_, cs);
        cudaStreamWaitEvent(v_.sh_cs_, v_.ev_fork_, 0);
        PdlScope pz(pdl, true);
        dense_proj(2, wg->native_type, wg->native_data, v_.xq_, v_.z_, (int) N, (int) ZV, n, v_.sh_cs_);
        if (v_.tp_diag_bits_ != nullptr && l < 3 ) {
            basalt::kernels::diag_copy_i32((const int32_t*) v_.z_, v_.tp_diag_bits_ + l * 288 + 72, 8, v_.sh_cs_);
        }
        cudaEventRecord(v_.ev_join_, v_.sh_cs_);
    };
    if (!read_q8) native_quantize_q8_1(xm, v_.xq_, (int) N, n, cs);
    // qkv and z in one dense_tc_pair launch for every window width, then the
    // one-launch recurrence with the gated norm
    if (gdnf && !pair_ok)
        note_fallback("GDN qkv + z without dense_tc_pair", wqkv->native_type,
                      wg->native_type);
    if (gdnf && pair_ok) {
        {
            PdlScope pp(pdl);   // after the hc read or the q8_1 kernel
            kernels::dense_tc_pair(wqkv->native_type, wqkv->native_data, qkv, (int) C,
                                   wg->native_type, wg->native_data, v_.z_, (int) ZV, v_.xq_, (int) N, n, cs);
            if (v_.tp_diag_bits_ != nullptr && l < 3 ) {   // v10: the card's qkv + z heads
                basalt::kernels::diag_copy_i32((const int32_t*) qkv, v_.tp_diag_bits_ + l * 288 + 64, 8, cs);
                basalt::kernels::diag_copy_i32((const int32_t*) v_.z_, v_.tp_diag_bits_ + l * 288 + 72, 8, cs);
            }
        }
        for (int i = 2; i <= 5; ++i) stamp(l, i);
        DecRecArgs ra;
        ra.state = state; ra.qkv = qkv; ra.conv_w = (const float*) wc->data; ra.x = xm;
        ra.w_alpha = w_alpha_p; ra.w_beta = w_beta_p;
        ra.dt = dt_p; ra.ssm_a = ssm_a_p; ra.z = v_.z_;
        ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = v_.xq_;
        ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
        if (v_.tp_) ra.y16 = v_.y16_;   // the pair's card keeps its heads' bf16 rows for the exchange
        ra.early_trigger = pdl;   // PDL2: the out projection stages its weights
        {
            PdlScope pr(pdl);
            rec(ra);
        }
        stamp(l, 6);
        if (v_.tp_ && !tp_assemble(ZV, ZVF, v_.y16_, true, (int) __LINE__, true)) return false;   // GDN head order
        if (v_.tp_diag_bits_ != nullptr && l < 3 ) {   // v10: the out-proj table head, both serves
            basalt::kernels::diag_copy_i32((const int32_t*) wout->native_data, v_.tp_diag_bits_ + l * 288 + 96, 8, cs);
        }
        if (v_.tp_ && v_.tp_diag_bits_ != nullptr && l < 3 ) {   // v10: the assembled halves
            basalt::kernels::diag_copy_i32((const int32_t*) v_.mixf_, v_.tp_diag_bits_ + l * 288 + 80, 8, cs);
            basalt::kernels::diag_copy_i32((const int32_t*) v_.mixf_ + ZV, v_.tp_diag_bits_ + l * 288 + 88, 8, cs);
        }
        if (v_.tp_ && v_.tp_diag_bits_ != nullptr && l < 3  && v_.y16_ != nullptr) {   // v13: the card's OWN y, pre-assembly
            basalt::kernels::diag_copy_i32((const int32_t*) v_.y16_, v_.tp_diag_bits_ + l * 288 + 112, 4, cs);
            basalt::kernels::diag_copy_i32((const int32_t*) v_.y16_ + 20 * 64, v_.tp_diag_bits_ + l * 288 + 116, 4, cs);
            basalt::kernels::diag_copy_i32((const int32_t*) v_.y16_ + 8 * 64, v_.tp_diag_bits_ + l * 288 + 120, 4, cs);
        }
        dense_proj(4, wout->native_type, wout->native_data, v_.xq_, v_.bo_, (int) ZVF, (int) N, n, cs);
        if (v_.tp_diag_bits_ != nullptr && l < 3 ) {
            basalt::kernels::diag_copy_i32((const int32_t*) wout->native_data, v_.tp_diag_bits_ + l * 288 + 96, 8, cs);
            basalt::kernels::diag_copy_i32((const int32_t*) v_.xq_, v_.tp_diag_bits_ + l * 288 + 104, 8, cs);
                   basalt::kernels::diag_copy_i32((const int32_t*) v_.bo_, v_.tp_diag_bits_ + l * 288 + 120, 8, cs);
            basalt::kernels::diag_copy_i32((const int32_t*) wout->native_data + 1105920 + 256, v_.tp_diag_bits_ + l * 288 + 56, 4, cs);
            basalt::kernels::diag_copy_i32((const int32_t*) v_.xq_ + 432, v_.tp_diag_bits_ + l * 288 + 60, 4, cs);
            if (v_.tp_diag_h1_ != nullptr) {   // v17: per-head xq hashes [0..24), table hashes [24..64), ptr delta [64..65]
                basalt::kernels::diag_hash_u8((const uint8_t*) v_.xq_, 144, 48, v_.tp_diag_h1_ + l * 96, cs);
                if (v_.tp_diag_bits_ != nullptr) { const int32_t* xq=(const int32_t*) v_.xq_;
                  basalt::kernels::diag_copy_i32(xq+0,    v_.tp_diag_bits_+l*288+128, 36, cs);
                  basalt::kernels::diag_copy_i32(xq+288,  v_.tp_diag_bits_+l*288+164, 36, cs);
                  basalt::kernels::diag_copy_i32(xq+864,  v_.tp_diag_bits_+l*288+200, 36, cs);
                  basalt::kernels::diag_copy_i32(xq+1152, v_.tp_diag_bits_+l*288+236, 36, cs);
                }
                basalt::kernels::diag_hash_u8((const uint8_t*) wout->native_data, 64 * 5040, 40, v_.tp_diag_h1_ + l * 96 + 48, cs);
                int32_t pd[2] = {(int32_t) (uintptr_t) wout->native_data, (int32_t) ((uintptr_t) wout->native_data >> 32)};
                cudaMemcpyAsync(v_.tp_diag_h1_ + l * 96 + 88, pd, 8, cudaMemcpyHostToDevice, cs);
            }
        }
        goto gdn_done;
    }
    const bool pk_z = gdnf_fork && v_.sh_hi_ != nullptr && n <= 2;
    if (pk_z) {   // z at the greatest priority beside qkv, then the one-launch recurrence
        cudaEventRecord(v_.ev_fork_, cs);
        cudaStreamWaitEvent(v_.sh_hi_, v_.ev_fork_, 0);
        {
            PdlScope pz(pdl, true);
            dense_proj(2, wg->native_type, wg->native_data, v_.xq_, v_.z_, (int) N, (int) ZV, n, v_.sh_hi_);
            if (v_.tp_diag_bits_ != nullptr && l < 3 ) {
                basalt::kernels::diag_copy_i32((const int32_t*) v_.z_, v_.tp_diag_bits_ + l * 288 + 72, 8, v_.sh_hi_);
            }
        }
        cudaEventRecord(v_.ev_join_, v_.sh_hi_);
        dense_proj(1, wqkv->native_type, wqkv->native_data, v_.xq_, qkv, (int) N, (int) C, n, cs);
        if (v_.tp_diag_bits_ != nullptr && l < 3 ) {
            basalt::kernels::diag_copy_i32((const int32_t*) qkv, v_.tp_diag_bits_ + l * 288 + 64, 8, cs);
        }
        stamp(l, 2);
        DecRecArgs ra;
        ra.state = state; ra.qkv = qkv; ra.conv_w = (const float*) wc->data; ra.x = xm;
        ra.w_alpha = w_alpha_p; ra.w_beta = w_beta_p;
        ra.dt = dt_p; ra.ssm_a = ssm_a_p; ra.z = v_.z_;
        ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = v_.xq_;
        ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
        if (v_.tp_) ra.y16 = v_.y16_;   // the pair's card keeps its heads' bf16 rows for the exchange
        cudaStreamWaitEvent(cs, v_.ev_join_, 0);
        for (int i = 3; i <= 5; ++i) stamp(l, i);
        {
            PdlScope pr(pdl, true);   // after the join
            dec_gdn_rec(ra, n, cs);
        }
        stamp(l, 6);
        if (v_.tp_ && !tp_assemble(ZV, ZVF, v_.y16_, true, (int) __LINE__, true)) return false;   // GDN head order
        dense_proj(4, wout->native_type, wout->native_data, v_.xq_, v_.bo_, (int) ZVF, (int) N, n, cs);
        if (v_.tp_diag_bits_ != nullptr && l < 3 ) {
            basalt::kernels::diag_copy_i32((const int32_t*) wout->native_data, v_.tp_diag_bits_ + l * 288 + 96, 8, cs);
            basalt::kernels::diag_copy_i32((const int32_t*) v_.xq_, v_.tp_diag_bits_ + l * 288 + 104, 8, cs);
                   basalt::kernels::diag_copy_i32((const int32_t*) v_.bo_, v_.tp_diag_bits_ + l * 288 + 120, 8, cs);
            basalt::kernels::diag_copy_i32((const int32_t*) wout->native_data + 1105920 + 256, v_.tp_diag_bits_ + l * 288 + 56, 4, cs);
            basalt::kernels::diag_copy_i32((const int32_t*) v_.xq_ + 432, v_.tp_diag_bits_ + l * 288 + 60, 4, cs);
            if (v_.tp_diag_h1_ != nullptr) {   // v17: per-head xq hashes [0..24), table hashes [24..64), ptr delta [64..65]
                basalt::kernels::diag_hash_u8((const uint8_t*) v_.xq_, 144, 48, v_.tp_diag_h1_ + l * 96, cs);
                if (v_.tp_diag_bits_ != nullptr) { const int32_t* xq=(const int32_t*) v_.xq_;
                  basalt::kernels::diag_copy_i32(xq+0,    v_.tp_diag_bits_+l*288+128, 36, cs);
                  basalt::kernels::diag_copy_i32(xq+288,  v_.tp_diag_bits_+l*288+164, 36, cs);
                  basalt::kernels::diag_copy_i32(xq+864,  v_.tp_diag_bits_+l*288+200, 36, cs);
                  basalt::kernels::diag_copy_i32(xq+1152, v_.tp_diag_bits_+l*288+236, 36, cs);
                }
                basalt::kernels::diag_hash_u8((const uint8_t*) wout->native_data, 64 * 5040, 40, v_.tp_diag_h1_ + l * 96 + 48, cs);
                int32_t pd[2] = {(int32_t) (uintptr_t) wout->native_data, (int32_t) ((uintptr_t) wout->native_data >> 32)};
                cudaMemcpyAsync(v_.tp_diag_h1_ + l * 96 + 88, pd, 8, cudaMemcpyHostToDevice, cs);
            }
        }
        goto gdn_done;
    }
    if (gdnf_fork && gdnf == 3) gdnf_z_side();
    dense_proj(1, wqkv->native_type, wqkv->native_data, v_.xq_, qkv, (int) N, (int) C, n, cs);
    if (v_.tp_diag_bits_ != nullptr && l < 3 ) {
        basalt::kernels::diag_copy_i32((const int32_t*) qkv, v_.tp_diag_bits_ + l * 288 + 64, 8, cs);
    }
    stamp(l, 2);
    if (gdnf) {
        DecRecArgs ra;
        ra.state = state; ra.qkv = qkv; ra.conv_w = (const float*) wc->data; ra.x = xm;
        ra.w_alpha = w_alpha_p; ra.w_beta = w_beta_p;
        ra.dt = dt_p; ra.ssm_a = ssm_a_p; ra.z = v_.z_;
        ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = v_.xq_;
        ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
        if (v_.tp_) ra.y16 = v_.y16_;   // the pair's card keeps its heads' bf16 rows for the exchange
        ra.early_trigger = pdl;   // PDL2: the out projection stages its weights
        if (gdnf_fork) {
            if (gdnf == 2) gdnf_z_side();
            ra.o = v_.y_;
            dec_gdn_rec_split(ra, n, cs);
            for (int i = 3; i <= 5; ++i) stamp(l, i);
            cudaStreamWaitEvent(cs, v_.ev_join_, 0);
            PdlScope pn(pdl, true);   // the norm follows the join
            dec_gdn_rec_norm(ra, n, cs);
            stamp(l, 6);
            if (v_.tp_ && !tp_assemble(ZV, ZVF, v_.y16_, true, (int) __LINE__, true)) return false;   // GDN head order
            dense_proj(4, wout->native_type, wout->native_data, v_.xq_, v_.bo_, (int) ZVF, (int) N, n, cs);
            if (v_.tp_diag_bits_ != nullptr && l < 3 ) {
                basalt::kernels::diag_copy_i32((const int32_t*) wout->native_data, v_.tp_diag_bits_ + l * 288 + 96, 8, cs);
                basalt::kernels::diag_copy_i32((const int32_t*) v_.xq_, v_.tp_diag_bits_ + l * 288 + 104, 8, cs);
            }
        } else {
            dense_proj(2, wg->native_type, wg->native_data, v_.xq_, v_.z_, (int) N, (int) ZV, n, cs);
            if (v_.tp_diag_bits_ != nullptr && l < 3 ) {
                basalt::kernels::diag_copy_i32((const int32_t*) v_.z_, v_.tp_diag_bits_ + l * 288 + 72, 8, cs);
            }
            for (int i = 3; i <= 5; ++i) stamp(l, i);
            dec_gdn_rec(ra, n, cs);
            stamp(l, 6);
            if (v_.tp_ && !tp_assemble(ZV, ZVF, v_.y16_, true, (int) __LINE__, true)) return false;   // GDN head order
            dense_proj(4, wout->native_type, wout->native_data, v_.xq_, v_.bo_, (int) ZVF, (int) N, n, cs);
            if (v_.tp_diag_bits_ != nullptr && l < 3 ) {
                basalt::kernels::diag_copy_i32((const int32_t*) wout->native_data, v_.tp_diag_bits_ + l * 288 + 96, 8, cs);
                basalt::kernels::diag_copy_i32((const int32_t*) v_.xq_, v_.tp_diag_bits_ + l * 288 + 104, 8, cs);
            }
        }
        goto gdn_done;
    }
    }
    if (v_.batch_rec_) {   // each row from its segment's conv history, the slots read from the table
        gdn_conv_l2_seg(v_.segb_, (size_t) gi * gdn_floats + (size_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size,
                        qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs);
    } else
    gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, 0);
    stamp(l, 3);
    gdn_ab_multi(xm, w_alpha_p, w_beta_p, dt_p,
                 ssm_a_p, gate, beta, (int) N, (int) HV,
                 n, cs);
    stamp(l, 4);
    dense_proj(2, wg->native_type, wg->native_data, v_.xq_, v_.z_, (int) N, (int) ZV, n, cs);
    if (v_.tp_diag_bits_ != nullptr && l < 3 ) {
        basalt::kernels::diag_copy_i32((const int32_t*) v_.z_, v_.tp_diag_bits_ + l * 288 + 72, 8, cs);
    }
    stamp(l, 5);
    // the recurrence from the untouched state over tokens [0, T); outputs only for this group's
    if (v_.batch_rec_) {   // each segment's recurrence from its own slot's state
        gdn_step_norm_seg(v_.segb_, (size_t) gi * gdn_floats, hb, (int) C, gate, beta, v_.z_, (const float*) wnm->data,
                          EPS, v_.y_, (int) HK, (int) HV, n, cs);
    } else
    gdn_step_norm_multi(state, hb, (int) C, gate, beta, v_.z_, (const float*) wnm->data, EPS, v_.y_, (int) HK,
                        (int) HV, T, nullptr, cs, 0);
    stamp(l, 6);
    if (v_.tp_) { if (!tp_assemble(ZV, ZVF, v_.y_, false, (int) __LINE__, true)) return false; }   // fp32 y rows placed directly, GDN head order
    else native_quantize_q8_1(v_.y_, v_.xq_, (int) ZV, n, cs);
    dense_proj(4, wout->native_type, wout->native_data, v_.xq_, v_.bo_, (int) ZVF, (int) N, n, cs);
    if (v_.tp_diag_bits_ != nullptr && l < 3 ) {
        basalt::kernels::diag_copy_i32((const int32_t*) wout->native_data, v_.tp_diag_bits_ + l * 288 + 96, 8, cs);
        basalt::kernels::diag_copy_i32((const int32_t*) v_.xq_, v_.tp_diag_bits_ + l * 288 + 104, 8, cs);
    }
    gdn_done:;
    return true;
}

}  // namespace basalt::core
