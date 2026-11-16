// src/models/qwen35/adapter.cpp - the Qwen3.5-family dense hybrid (Qwen3.8-27B): registers the architecture, reads its
// pack into the shared dense stack and hands out its sessions.
#include "models/qwen35/adapter.hpp"

#include "models/qwen35/shard.hpp"

#include "basalt/core/diag.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/model/rope_flags.hpp"
#include "basalt/program/options.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/program/qwen_forms.hpp"
#include "models/qwen35/meta.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <optional>

namespace basalt::models::qwen35 {
namespace {

// the pack's DFlash 2 block drafter's shape (qwen35.dflash.* metadata)
bool block_meta(const GgufFile& gg, dense::BlockDrafter& d, int H, std::string& err) {
    const auto u = [&](const char* k) { const MetaValue* v = gg.get(std::string("qwen35.dflash.") + k); return v ? (int) v->u : 0; };
    const auto f = [&](const char* k) { const MetaValue* v = gg.get(std::string("qwen35.dflash.") + k); return v ? (float) v->num() : 0.0f; };
    d.block = u("block_size");
    d.mask_id = u("mask_token_id");
    d.conv_group = u("conv_group");
    d.rank = u("selector_rank");
    d.top_k = u("selector_top_k");
    d.window = u("sliding_window");
    d.n_head = u("attention.head_count");
    d.n_kv = u("attention.head_count_kv");
    d.head_dim = u("attention.key_length");
    d.theta = f("rope.freq_base");
    d.eps = f("attention.layer_norm_rms_epsilon");
    if (const MetaValue* t = gg.get("qwen35.dflash.target_layers"))
        for (const MetaValue& x : t->items) d.target_layers.push_back((int) x.u);
    if (d.block < 2 || d.block > 8 || d.target_layers.empty() || u("conv_kernel") != 2 || d.n_kv <= 0 ||
        (d.head_dim != 128 && d.head_dim != 256) || d.top_k > 20 || H % d.conv_group) {
        err = "qwen35: the pack's block drafter has a shape the engine does not run";
        return false;
    }
    for (int l = 0; l < u("block_count"); ++l) d.layers.emplace_back();
    d.present = true;
    return true;
}


// The device bytes a weight takes: its payload in its card's weight block, 256-byte aligned (load_weights carves every
// tensor from one allocation per card, so the planner counts exactly what the load takes).
uint64_t weight_bytes(const TensorInfo& t) { return (tensor_payload_bytes(t) + 255) & ~(uint64_t) 255; }

// as weight_bytes, for the bytes a tensor-parallel card holds of it
uint64_t shard_weight_bytes(uint64_t payload) { return (payload + 255) & ~(uint64_t) 255; }

// Whether the load places tensor `name` on a card: the layers [lo, hi), and with lo == 0 the head and the drafter in use.
// The drafter is one card's (its layers run whole there); the head is every card's, each its own shard of it.
bool on_card(const std::string& name, const dense::Options& o, int lo, int hi, int rank = 0) {
    if (name.starts_with("blk.")) {
        const int il = std::atoi(name.c_str() + 4);
        return il >= lo && il < hi;
    }
    return lo == 0 && (name.starts_with("output") ||
                       (rank == 0 && ((o.drafter == dense::Drafter::Block && name.starts_with("dflash.")) ||
                                      (o.drafter == dense::Drafter::Mtp && name.starts_with("mtp.")))));
}

// The dense stack's options for a run of these flags (the drafter the pack carries, when drafting).
dense::Options run_options(const GgufFile& gg, int64_t max_context, int sessions, int spec, bool draft, bool vision, const std::string& kv,
                           int64_t kv_resident) {
    const auto has = [&](const char* name) {
        for (const TensorInfo& t : gg.tensors())
            if (t.name == name) return true;
        return false;
    };
    const bool block = draft && has("dflash.fc.0.weight"), mtp = draft && !block && has("mtp.fc.weight");
    dense::Options o;
    o.max_context = max_context;
    o.sessions = sessions;
    o.spec = spec;
    o.draft = draft;
    o.drafter = block ? dense::Drafter::Block : mtp ? dense::Drafter::Mtp : dense::Drafter::None;
    o.vision = vision;
    o.kv_int8 = kv == "int8";
    o.kv_resident = kv_resident;
    o.arch = "qwen35";
    return o;
}

// Where the layers go with a second card: all on the main card when the model, its cells and scratch fit there (the second card
// then reads long prompts), else split across the two (dense::plan_split), from the pack's bytes per layer and both cards'
// free memory (CUDA device 0 and `peer`).  `forced`: a cut taken as given.  `tp`: price every card holding half of every matrix
// instead of a layer cut, with which half of the fixed bytes the pair shares out (the head's).
dense::SplitPlan plan_pack(const GgufFile& gg, const dense::Geometry& g, const dense::Options& o, const dense::BlockDrafter& bd, int peer,
                           double main_gbps, double peer_gbps, int reserve_mib, int forced, int tp = 0) {
    std::vector<uint64_t> layer((size_t) g.n_layers, 0);
    uint64_t main_weights = 0, head_weights = 0;
    for (const TensorInfo& t : gg.tensors()) {
        if (t.name.starts_with("blk.")) {
            const int il = std::atoi(t.name.c_str() + 4);
            if (il >= 0 && il < g.n_layers) layer[(size_t) il] += weight_bytes(t);
        } else if (on_card(t.name, o, 0, 0)) {
            main_weights += weight_bytes(t);
            if (t.name.starts_with("output.")) head_weights += weight_bytes(t);   // the vocabulary: both cards run part of it
        }
    }
    size_t main_free = 0, peer_free = 0, total = 0;
    {
        const core::OnDevice on(0);
        cudaMemGetInfo(&main_free, &total);
    }
    {
        const core::OnDevice on(peer);
        cudaMemGetInfo(&peer_free, &total);
    }
    constexpr uint64_t kMargin = 512ull << 20;
    const uint64_t reserve = ((uint64_t) std::max(reserve_mib, 0) << 20) + kMargin;
    const dense::SplitCard mc{main_free > reserve ? main_free - reserve : 0, main_gbps};
    const dense::SplitCard pc{peer_free > kMargin ? peer_free - kMargin : 0, peer_gbps};
    dense::SplitInputs in = dense::split_inputs(g, o, bd, layer, main_weights, mc, pc, head_weights);
    in.forced = forced;
    in.tensor_parallel = tp > 0;
    return dense::plan_split(in);
}

class Loaded final : public model::LoadedModel {
public:
    ~Loaded() override {
        model_.reset();
        for (auto& [dev, b] : blocks_) cudaFree(b.base);
        if (embd_) cudaFreeHost(embd_);
        if (fd_ >= 0) platform::close_fd(fd_);
    }

    const model::ModelDesc& desc() const override { return desc_; }

    model::Result make_session(std::unique_ptr<model::ModelSession>& out) override {
        std::string err;
        out = dense::make_session(*model_, err);
        if (!out) return {model::Status::Error, err};
        return {};
    }

    bool load(const model::LoadRequest& req, std::string& err) {
        const std::string& path = req.paths.front();
        const GgufFile gg(path);
        desc_ = describe_meta(gg);
        dense::Geometry g = geometry(gg, err);
        if (!err.empty()) return false;
        // the rope scaling of the run (rope_flags.hpp): the model's own base and trained context, the pack's keys, the flags
        g.rope.scaling.freq_base = g.rope.theta;
        g.rope.scaling.orig_ctx = (double) (g.context_length > 0 ? g.context_length : 262144);
        if (!model::resolve_rope_scaling(req.rope, model::rope_keys(gg, "qwen35"), req.max_context, "basalt qwen35", g.rope.scaling,
                                         err)) {
            err = "qwen35: " + err;
            return false;
        }
        g.rope.theta = (float) g.rope.scaling.freq_base;
        // the longest context: the trained one, or as far as the scaling stretches it
        const int64_t longest = g.rope.scaling.type == kernels::RopeScalingType::None
                                    ? g.context_length
                                    : (int64_t) (g.rope.scaling.orig_ctx / g.rope.scaling.freq_scale());
        if (longest > 0 && req.max_context > longest) {
            err = "qwen35: --max-context " + std::to_string((long long) req.max_context) + " is beyond the model's " +
                  std::to_string((long long) longest) +
                  (g.rope.scaling.type == kernels::RopeScalingType::None ? " (--rope-scaling yarn --rope-scale F extends it)"
                                                                         : " with this rope scaling");
            return false;
        }
        for (const TensorInfo& t : gg.tensors()) tensors_[t.name] = t;
        geom_ = g;
        dense::Options o = run_options(gg, req.max_context, req.sessions, req.spec, req.draft, req.vision, req.kv, req.kv_resident);
        const bool want_block = o.drafter == dense::Drafter::Block, want_mtp = o.drafter == dense::Drafter::Mtp;
        dense::BlockDrafter bd_meta;
        if (want_block && !block_meta(gg, bd_meta, g.n_embd, err)) return false;
        if (!req.peer_devices.empty() && !plan_cards(gg, g, o, bd_meta, req, err)) return false;
        fd_ = platform::open_fd_utf8(path);
        if (fd_ < 0) {
            err = "cannot open " + path;
            return false;
        }
        uint64_t total = 0;
        for (const TensorInfo& t : gg.tensors())
            if (!t.name.starts_with("mtp.")) total += tensor_payload_bytes(t);
        // a layer split's later layers count on the second card
        uint64_t peer_total = 0;
        for (const TensorInfo& t : gg.tensors())
            if (split_at_ > 0 && t.name.starts_with("blk.") && std::atoi(t.name.c_str() + 4) >= split_at_) peer_total += tensor_payload_bytes(t);
        // a tensor-parallel pair counts what each card reads: its shard of a split tensor, the whole of a shared one
        if (tp_ > 0) {
            total = peer_total = 0;
            for (const TensorInfo& t : gg.tensors()) {
                if (t.name.starts_with("mtp.")) continue;   // the drafter layer joins the count with --spec, as before
                if (t.name == "token_embd.weight") {        // page-locked RAM, read once for both cards
                    total += tensor_payload_bytes(t);
                    continue;
                }
                if (on_card(t.name, o, 0, g.n_layers, 0)) {
                    const uint64_t n = card_payload(t, 0, err);
                    if (!n) return false;
                    total += n;
                }
                if (on_card(t.name, o, 0, g.n_layers, 1)) {
                    const uint64_t n = card_payload(t, 1, err);
                    if (!n) return false;
                    peer_total += n;
                }
            }
            total += peer_total;   // as in the other paths, `total` is what both cards read
        }
        core::LoadProgress::announce_total(total);
        core::LoadProgress::plan(core::LoadProgress::cuda(0), total - peer_total);
        if (peer_total) core::LoadProgress::plan(core::LoadProgress::cuda(req.peer_devices.front()), peer_total);
        core::LoadProgress progress(core::LoadProgress::cuda(0), total - peer_total);
        std::optional<core::LoadProgress> peer_progress;
        if (peer_total) peer_progress.emplace(core::LoadProgress::cuda(req.peer_devices.front()), peer_total);
        progress_ = &progress;
        {   // each card's weights in one block: the main card's, and a split's later layers on the second card
            const int end = split_at_ > 0 ? split_at_ : g.n_layers;
            if (!reserve_block(gg, o, 0, end, err)) return false;
            if (split_at_ > 0) {
                const core::OnDevice on(req.peer_devices.front());
                if (!reserve_block(gg, o, split_at_, g.n_layers, err)) return false;
            } else if (tp_ > 0) {   // the pair's second card: its own block, its own shards
                const core::OnDevice on(req.peer_devices.front());
                if (!reserve_block(gg, o, 0, g.n_layers, err, 1)) return false;
            }
        }
        dense::Weights w;
        const int H = g.n_embd;
        // the token embedding stays in page-locked host memory; the kernels read the rows they need
        const TensorInfo* te = need(gg, "token_embd.weight", kTypeBf16, (int64_t) H * g.n_vocab, err);
        if (!te) return false;
        if (cudaHostAlloc(&embd_, tensor_payload_bytes(*te), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            err = "out of page-locked memory for the token embedding";
            return false;
        }
        if (!platform::pread_all(fd_, embd_, tensor_payload_bytes(*te), gg.data_start() + te->offset)) {
            err = "short read of token_embd.weight";
            return false;
        }
        progress.add(tensor_payload_bytes(*te));
        void* dev_embd = nullptr;
        cudaHostGetDevicePointer(&dev_embd, embd_, 0);
        w.embd = (const uint16_t*) dev_embd;
        w.out_norm = floats(gg, "output_norm.weight", H, err);
        if (!w.out_norm || !linear(gg, "output", g.n_vocab, H, w.head, err)) return false;
        const int AQ = g.n_head * g.head_dim, KV = g.n_kv * g.head_dim;
        // a layer split's later layers go to the second card (their own weights there; the main card's entries empty)
        dense::Weights peer_w;
        for (int il = 0; il < g.n_layers; ++il) {
            dense::Layer L, P;
            L.kind = P.kind = g.kinds[(size_t) il];
            if (split_at_ > 0 && il >= split_at_) {
                const uint64_t before = dev_bytes_;
                const core::OnDevice on(req.peer_devices.front());
                progress_ = &*peer_progress;
                const bool ok = layer(gg, g, il, P, err);
                progress_ = &progress;
                if (!ok) return false;
                peer_bytes_ += dev_bytes_ - before;
                dev_bytes_ = before;
            } else if (!layer(gg, g, il, L, err)) {
                return false;
            }
            w.layers.push_back(L);
            if (tp_ == 0) peer_w.layers.push_back(P);   // the pair's layers are read together, on the second card
        }
        if (tp_ > 0) {   // the pair's second card, in its own block on its own device: its shards of the head and of every layer
            const uint64_t before = dev_bytes_;
            bool ok = true;
            {
                const core::OnDevice on(req.peer_devices.front());
                progress_ = &*peer_progress;
                peer_w.out_norm = floats(gg, "output_norm.weight", H, err, 1);
                ok = peer_w.out_norm && linear(gg, "output", g.n_vocab, H, peer_w.head, err, 1);
                for (int il = 0; ok && il < g.n_layers; ++il) {
                    dense::Layer P;
                    P.kind = g.kinds[(size_t) il];
                    ok = layer(gg, g, il, P, err, 1);
                    peer_w.layers.push_back(P);
                }
                progress_ = &progress;
            }
            if (!ok) return false;
            peer_bytes_ += dev_bytes_ - before;
            dev_bytes_ = before;
        }
        if (want_block && !block_drafter(gg, w.dflash, H, err)) return false;
        if (want_mtp) {   // the pack's drafter layer, for --spec
            dense::Mtp& d = w.mtp;
            dense::Layer& L = d.layer;
            L.kind = dense::Mixer::Attention;
            d.norm_e = floats(gg, "mtp.pre_fc_norm_embedding.weight", H, err);
            d.norm_h = floats(gg, "mtp.pre_fc_norm_hidden.weight", H, err);
            d.norm = floats(gg, "mtp.norm.weight", H, err);
            L.attn_norm = floats(gg, "mtp.attn_norm.weight", H, err);
            L.ffn_norm = floats(gg, "mtp.post_attention_norm.weight", H, err);
            L.q_norm = floats(gg, "mtp.attn_q_norm.weight", g.head_dim, err);
            L.k_norm = floats(gg, "mtp.attn_k_norm.weight", g.head_dim, err);
            if (!d.norm_e || !d.norm_h || !d.norm || !L.attn_norm || !L.ffn_norm || !L.q_norm || !L.k_norm ||
                !linear(gg, "mtp.fc", H, 2 * H, d.fc, err) || !linear(gg, "mtp.attn_q", 2 * AQ, H, L.q, err) ||
                !linear(gg, "mtp.attn_k", KV, H, L.k, err) || !linear(gg, "mtp.attn_v", KV, H, L.v, err) ||
                !linear(gg, "mtp.attn_output", H, AQ, L.o, err) || !linear(gg, "mtp.ffn_gate", g.n_ff, H, L.gate, err) ||
                !linear(gg, "mtp.ffn_up", g.n_ff, H, L.up, err) || !linear(gg, "mtp.ffn_down", H, g.n_ff, L.down, err))
                return false;
            d.present = true;
        }
        progress.finish();
        if (peer_progress) peer_progress->finish();
        progress_ = nullptr;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        basalt::log::info("basalt qwen35", "%zu tensors, %.2f GiB on the GPU, %.2f GiB of embedding in page-locked RAM; "
                          "%.2f GiB free", n_tensors_, (double) dev_bytes_ / 1073741824.0, (double) tensor_payload_bytes(*te) / 1073741824.0,
                          (double) free_b / 1073741824.0);
        if (split_at_ > 0)
            basalt::log::info("basalt qwen35", "layer split: %.2f GiB of layers %d-%d on CUDA%d", (double) peer_bytes_ / 1073741824.0, split_at_,
                              g.n_layers - 1, req.peer_devices.front());
        if (tp_ > 0) {
            basalt::log::info("basalt qwen35", "tensor parallel %d: %.2f GiB on CUDA0, %.2f GiB on CUDA%d (half of every matrix, the "
                              "delta-net by key-head group)", tp_, (double) dev_bytes_ / 1073741824.0,
                              (double) peer_bytes_ / 1073741824.0, req.peer_devices.front());
            // The plan and both cards' bytes are in place; no product crosses the pair yet, so this load cannot serve.
            err = "qwen35: tensor parallel places both cards' shards, but the layers do not run split between the cards";
            return false;
        }
        o.host_embedding_bytes = tensor_payload_bytes(*te);
        o.gpu_weight_bytes = dev_bytes_ + peer_bytes_;
        o.layer_end = split_at_;
        if (o.vision) {
            core::ModelFile file;
            core::ModelSources srcs;
            if (!file.open(path, err)) return false;
            core::model_sources(file, srcs);
            o.mmproj = srcs.vision;
        }
        const uint64_t need_b = dense::shared_bytes(g, o.max_context, o.kv_int8, o.kv_resident) +
                                (uint64_t) o.sessions * dense::session_bytes(g, o.max_context, o.drafter == dense::Drafter::Mtp, o.kv_int8,
                                                                             o.kv_resident, 0, split_at_) +
                                (req.peer_devices.empty() ? 0
                                 : split_at_ > 0           ? dense::stage_main_bytes(g, w.dflash)
                                                           : dense::prompt_line_main_bytes(g, w.dflash, o.max_context));
        if (const uint64_t host_b = dense::session_host_bytes(g, o.max_context, o.kv_int8, o.kv_resident))
            basalt::log::info("basalt qwen35", "K/V streaming: %lld of %lld cells per attention layer in VRAM, %.2f GiB in page-locked "
                              "RAM per slot", (long long) dense::resident_cells(o.max_context, o.kv_resident), (long long) o.max_context,
                              (double) host_b / 1073741824.0);
        const uint64_t reserve = (uint64_t) std::max(req.vram_reserve_mib, 0) << 20;
        if (need_b + reserve + (512ull << 20) > free_b) {
            err = "qwen35: " + std::to_string((long long) o.sessions) + " session(s) of " + std::to_string((long long) o.max_context) +
                  " cells need " + std::to_string((unsigned long long) (need_b >> 20)) + " MiB, " +
                  std::to_string((unsigned long long) (free_b >> 20)) + " MiB are free; lower --max-context or --parallel";
            return false;
        }
        model_ = dense::make_model(g, w, o, err);
        if (!model_) return false;
        if (split_at_ > 0) return dense::attach_split(*model_, req.peer_devices.front(), split_at_, peer_w, err);
        if (!req.peer_devices.empty()) prompt_line(gg, g, req.peer_devices.front(), o.max_context, o.kv_resident);
        return true;
    }

private:
    // Where the layers go with a second card (plan_pack); --diag-split-at takes its cut as given, --diag-tp prices the
    // pair that holds half of every matrix instead.
    bool plan_cards(const GgufFile& gg, const dense::Geometry& g, const dense::Options& o, const dense::BlockDrafter& bd,
                    const model::LoadRequest& req, std::string& err) {
        const int peer = req.peer_devices.front();
        if (req.diag_tp_width > 0 && !tp_geometry(g, req.diag_tp_width, err)) return false;
        const dense::SplitPlan p = plan_pack(gg, g, o, bd, peer, req.card_gbps.size() > 0 ? req.card_gbps[0] : 0,
                                             req.card_gbps.size() > 1 ? req.card_gbps[1] : 0, req.vram_reserve_mib,
                                             req.diag_split_at, req.diag_tp_width);
        if (!p.error.empty()) {
            err = "qwen35: " + p.error + "; lower --max-context or --parallel";
            return false;
        }
        if (req.diag_tp_width > 0) {
            tp_ = req.diag_tp_width;
            basalt::log::info("basalt qwen35", "tensor parallel %d: %.2f GiB on CUDA0 and %.2f GiB on CUDA%d, %lld layers", tp_,
                              (double) p.tp_main / 1073741824.0, (double) p.tp_peer / 1073741824.0, peer, (long long) g.n_layers);
            return true;
        }
        split_at_ = p.cut;
        if (p.cut > 0)
            basalt::log::info("basalt qwen35", "layer split: layers 0-%d (%.2f GiB) on CUDA0, %d-%d (%.2f GiB) on CUDA%d", p.cut - 1,
                              (double) p.main_bytes / 1073741824.0, p.cut, g.n_layers - 1, (double) p.peer_bytes / 1073741824.0, peer);
        return true;
    }

    // A second card reads long prompts' last layers (dense::attach_prompt_line): their weights go to that card too.  Without
    // room for it there the prompts stay on the main card.
    void prompt_line(const GgufFile& gg, const dense::Geometry& g, int dev, int64_t max_context, int64_t kv_resident) {
        const int l_min = std::max(0, g.n_layers - dense::kPromptLineLayers);
        int main_dev = 0;
        cudaGetDevice(&main_dev);
        if (dense::resident_cells(max_context, kv_resident) < max_context) {   // the line copies cells between the cards' caches
            basalt::log::info("basalt qwen35", "prompt line off: K/V streaming (--kv-resident) keeps part of the cache in RAM");
            return;
        }
        if (dense::prompt_line_main_bytes(g, dense::BlockDrafter{}, max_context) == 0) {   // before the weights go over
            basalt::log::info("basalt qwen35", "prompt line off: a context of %lld holds no prompt that a second card would read",
                              (long long) max_context);
            return;
        }
        const uint64_t main_bytes = dev_bytes_;
        std::string err;
        dense::Weights lw;
        bool ok = cudaSetDevice(dev) == cudaSuccess;
        if (!ok) err = "qwen35: cannot use CUDA" + std::to_string(dev);
        dense::Options lo;
        ok = ok && reserve_block(gg, lo, l_min, g.n_layers, err);
        for (int il = 0; ok && il < g.n_layers; ++il) {
            dense::Layer L;
            L.kind = g.kinds[(size_t) il];
            if (il >= l_min) ok = layer(gg, g, il, L, err);
            lw.layers.push_back(L);
        }
        const uint64_t line_bytes = dev_bytes_ - main_bytes;
        dev_bytes_ = main_bytes;
        cudaSetDevice(main_dev);
        if (ok) basalt::log::info("basalt qwen35", "prompt line: layers %d-%d (%.2f GiB) on CUDA%d, read beside the main card's on CUDA%d", l_min,
                                  g.n_layers - 1, (double) line_bytes / 1073741824.0, dev, main_dev);
        if (ok && dense::attach_prompt_line(*model_, dev, l_min, lw, err)) return;
        basalt::log::warn("basalt qwen35", "prompt line off, prompts read on CUDA%d: %s", main_dev, err.c_str());
        if (auto it = blocks_.find(dev); it != blocks_.end()) {
            cudaFree(it->second.base);
            blocks_.erase(it);
        }
        (void) cudaGetLastError();
    }

    // layer il's weights, as card `rank` of the pair holds them (on the current device)
    bool layer(const GgufFile& gg, const dense::Geometry& g, int il, dense::Layer& L, std::string& err, int rank = 0) {
        const int H = g.n_embd;
        const int C = g.conv_channels(), VD = g.value_width(), HV = g.gdn_v_heads, AQ = g.n_head * g.head_dim;
        const int KV = g.n_kv * g.head_dim;
        const std::string p = "blk." + std::to_string(il) + ".";
        L.kind = g.kinds[(size_t) il];
        L.attn_norm = floats(gg, p + "attn_norm.weight", H, err, rank);
        L.ffn_norm = floats(gg, p + "post_attention_norm.weight", H, err, rank);
        if (!L.attn_norm || !L.ffn_norm) return false;
        if (L.kind == dense::Mixer::Gdn) {
            if (!linear(gg, p + "attn_qkv", C, H, L.qkv, err, rank) || !linear(gg, p + "attn_gate", VD, H, L.z, err, rank) ||
                !linear(gg, p + "ssm_out", H, VD, L.out, err, rank))
                return false;
            L.w_ab = bf16_pair(gg, p + "ssm_alpha.weight", p + "ssm_beta.weight", (int64_t) HV * H, err, rank);
            L.dt = floats(gg, p + "ssm_dt.bias", HV, err, rank);
            L.ssm_a = floats(gg, p + "ssm_a", HV, err, rank);
            L.conv = floats(gg, p + "ssm_conv1d.weight", (int64_t) C * g.gdn_conv, err, rank);
            L.gamma = floats(gg, p + "ssm_norm.weight", g.gdn_state, err, rank);
            if (!L.w_ab || !L.dt || !L.ssm_a || !L.conv || !L.gamma) return false;
        } else {
            if (!linear(gg, p + "attn_q", 2 * AQ, H, L.q, err, rank) || !linear(gg, p + "attn_k", KV, H, L.k, err, rank) ||
                !linear(gg, p + "attn_v", KV, H, L.v, err, rank) || !linear(gg, p + "attn_output", H, AQ, L.o, err, rank))
                return false;
            L.q_norm = floats(gg, p + "attn_q_norm.weight", g.head_dim, err, rank);
            L.k_norm = floats(gg, p + "attn_k_norm.weight", g.head_dim, err, rank);
            if (!L.q_norm || !L.k_norm) return false;
        }
        if (!linear(gg, p + "ffn_gate", g.n_ff, H, L.gate, err, rank) || !linear(gg, p + "ffn_up", g.n_ff, H, L.up, err, rank) ||
            !linear(gg, p + "ffn_down", H, g.n_ff, L.down, err, rank))
            return false;
        return true;
    }

    // the pack's DFlash 2 block drafter (dflash.* tensors, qwen35.dflash.* metadata)
    bool block_drafter(const GgufFile& gg, dense::BlockDrafter& d, int H, std::string& err) {
        if (!block_meta(gg, d, H, err)) return false;
        d.layers.clear();
        d.present = false;
        const auto u = [&](const char* k) { const MetaValue* v = gg.get(std::string("qwen35.dflash.") + k); return v ? (int) v->u : 0; };
        const int layers = u("block_count"), ff = u("feed_forward_length");
        const int AQ = d.n_head * d.head_dim, KV = d.n_kv * d.head_dim, conv = 2 * 2 * (H / d.conv_group);
        for (size_t j = 0; j < d.target_layers.size(); ++j) {
            d.fc.emplace_back();
            if (!linear(gg, "dflash.fc." + std::to_string(j), H, H, d.fc.back(), err)) return false;
        }
        d.hidden_norm = floats(gg, "dflash.hidden_norm.weight", H, err);
        d.norm = floats(gg, "dflash.norm.weight", H, err);
        if (!d.hidden_norm || !d.norm) return false;
        for (int l = 0; l < layers; ++l) {
            const std::string p = "dflash.blk." + std::to_string(l) + ".";
            dense::BlockDrafter::Layer L;
            L.attn_norm = floats(gg, p + "attn_norm.weight", H, err);
            L.ffn_norm = floats(gg, p + "post_attention_norm.weight", H, err);
            L.q_norm = floats(gg, p + "attn_q_norm.weight", d.head_dim, err);
            L.k_norm = floats(gg, p + "attn_k_norm.weight", d.head_dim, err);
            L.base_a = floats(gg, p + "attn_conv.base", 4 * (int64_t) H, err);
            L.base_f = floats(gg, p + "ffn_conv.base", 4 * (int64_t) H, err);
            if (!L.attn_norm || !L.ffn_norm || !L.q_norm || !L.k_norm || !L.base_a || !L.base_f ||
                !linear(gg, p + "attn_q", AQ, H, L.q, err) || !linear(gg, p + "attn_k", KV, H, L.k, err) ||
                !linear(gg, p + "attn_v", KV, H, L.v, err) || !linear(gg, p + "attn_output", H, AQ, L.o, err) ||
                !linear(gg, p + "ffn_gate", ff, H, L.gate, err) || !linear(gg, p + "ffn_up", ff, H, L.up, err) ||
                !linear(gg, p + "ffn_down", H, ff, L.down, err) || !linear(gg, p + "attn_conv", conv, H, L.conv_a, err) ||
                !linear(gg, p + "ffn_conv", conv, H, L.conv_f, err))
                return false;
            d.layers.push_back(L);
        }
        d.sel_h = bf16(gg, "dflash.selector.hidden.weight", (int64_t) d.rank * H, err);
        d.pred = bf16(gg, "dflash.selector.pred", (int64_t) d.rank * desc_.n_vocab, err);
        d.succ = bf16(gg, "dflash.selector.succ", (int64_t) d.rank * desc_.n_vocab, err);
        if (!d.sel_h || !d.pred || !d.succ) return false;
        d.present = true;
        return true;
    }

    const uint16_t* bf16(const GgufFile& gg, const std::string& name, int64_t elems, std::string& err, int rank = 0) {
        const TensorInfo* t = need(gg, name, kTypeBf16, elems, err);
        return t ? (const uint16_t*) upload(gg, *t, err, rank) : nullptr;
    }

    const TensorInfo* need(const GgufFile& gg, const std::string& name, int type, int64_t elems, std::string& err) {
        (void) gg;
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            err = "qwen35: the pack has no " + name;
            return nullptr;
        }
        const TensorInfo* t = &it->second;
        if ((int) t->type != type || (int64_t) t->elements() != elems) {
            err = "qwen35: " + name + " is not the expected type and size";
            return nullptr;
        }
        return t;
    }

    // The card's shard of `t`, whole-dimension runs and all (dim None when the split leaves the tensor alone).  The whole
    // matrix is the pack's declared shape (GGUF order: `shape[0]` is a row's width), so the rules check the pack's shapes.
    bool shard_of(const TensorInfo& t, int rank, dense::MatrixShard& out, std::string& err) const {
        if (tp_ == 0) return (out = dense::MatrixShard{}, true);
        const int64_t cols = t.shape.size() > 1 ? (int64_t) t.shape[0] : 1;
        return tp_shard(geom_, t.name, (int64_t) t.elements() / cols, cols, tp_, rank, out, err);
    }

    // the payload card `rank` holds of `t`: its shard's when the split cuts it, all of it otherwise
    uint64_t card_payload(const TensorInfo& t, int rank, std::string& err) const {
        dense::MatrixShard s;
        if (tp_ == 0) return tensor_payload_bytes(t);
        if (!shard_of(t, rank, s, err)) return 0;
        if (s.whole()) return tensor_payload_bytes(t);
        const int64_t n = dense::shard_bytes(t.type, s, err);
        if (n <= 0) { err = "qwen35: " + t.name + ": " + err; return 0; }
        return (uint64_t) n;
    }

    // as weight_bytes, for what the card holds
    uint64_t card_bytes(const TensorInfo& t, int rank, std::string& err) const {
        const uint64_t n = card_payload(t, rank, err);
        return n ? shard_weight_bytes(n) : 0;
    }

    // One allocation on the current device for the tensors the load places there (on_card's), carved by take().
    bool reserve_block(const GgufFile& gg, const dense::Options& o, int lo, int hi, std::string& err, int rank = 0) {
        uint64_t bytes = 0;
        for (const TensorInfo& t : gg.tensors()) {
            if (!on_card(t.name, o, lo, hi, rank)) continue;
            const uint64_t n = card_bytes(t, rank, err);
            if (!n) return false;
            bytes += n;
        }
        int dev = 0;
        cudaGetDevice(&dev);
        Block& b = blocks_[dev];
        if (cudaMalloc((void**) &b.base, bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            blocks_.erase(dev);
            err = "qwen35: out of device memory for " + std::to_string((unsigned long long) (bytes >> 20)) + " MiB of weights on CUDA" +
                  std::to_string(dev);
            return false;
        }
        b.size = bytes;
        return true;
    }
    // `bytes` of the current device's block (256-byte aligned)
    void* take(uint64_t bytes, const std::string& name, std::string& err) {
        int dev = 0;
        cudaGetDevice(&dev);
        const auto it = blocks_.find(dev);
        const uint64_t need = (bytes + 255) & ~(uint64_t) 255;
        if (it == blocks_.end() || it->second.used + need > it->second.size) {
            err = "qwen35: " + name + " does not fit the weights' block on CUDA" + std::to_string(dev);
            return nullptr;
        }
        void* d = it->second.base + it->second.used;
        it->second.used += need;
        ++n_tensors_;
        return d;
    }

    void* upload(const GgufFile& gg, const TensorInfo& t, std::string& err, int rank = 0) {
        const uint64_t n = card_payload(t, rank, err);
        if (!n) return nullptr;
        void* d = take(n, t.name, err);
        if (!d) return nullptr;
        if (!place(gg, t, rank, (uint8_t*) d, err)) return nullptr;
        dev_bytes_ += n;
        return d;
    }

    // the card's bytes of `t` at `dst`: a whole tensor read in chunks, or its shard, whose runs the pack's layout
    // scattered - read the tensor once and cut the card's copy here, or read the runs straight from the file when the
    // tensor is too big to hold twice
    bool place(const GgufFile& gg, const TensorInfo& t, int rank, uint8_t* dst, std::string& err) {
        dense::MatrixShard s;
        if (!shard_of(t, rank, s, err)) return false;
        constexpr uint64_t kChunk = 64ull << 20, kTwice = 128ull << 20;
        std::vector<dense::ByteSpan> spans;
        if (!s.whole()) {
            if (!dense::shard_spans(t.type, s, spans, err)) {
                err = "qwen35: " + t.name + ": " + err;
                return false;
            }
        } else {
            spans.push_back({0, 0, (int64_t) tensor_payload_bytes(t)});
        }
        uint64_t card = 0;
        for (const dense::ByteSpan& sp : spans) card += (uint64_t) sp.n;
        if (spans.size() > 8 && tensor_payload_bytes(t) <= kTwice) {
            whole_.resize((size_t) tensor_payload_bytes(t));
            cut_.resize((size_t) card);
            if (!platform::pread_all(fd_, whole_.data(), tensor_payload_bytes(t), gg.data_start() + t.offset)) {
                err = "qwen35: reading " + t.name + " failed";
                return false;
            }
            if (progress_) progress_->add(card);   // the card's bytes, as every path counts (this one read the tensor whole)
            dense::gather_spans(whole_.data(), spans, cut_.data());
            if (cudaMemcpy(dst, cut_.data(), (size_t) card, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "qwen35: placing " + t.name + " failed";
                return false;
            }
            return true;
        }
        for (const dense::ByteSpan& sp : spans) {
            staging_.resize((size_t) std::min<uint64_t>((uint64_t) sp.n, kChunk));
            for (uint64_t off = 0; off < (uint64_t) sp.n; off += kChunk) {
                const uint64_t n = std::min<uint64_t>(kChunk, (uint64_t) sp.n - off);
                if (!platform::pread_all(fd_, staging_.data(), n, gg.data_start() + t.offset + (uint64_t) sp.src + off) ||
                    cudaMemcpy(dst + sp.dst + off, staging_.data(), n, cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "qwen35: reading " + t.name + " failed";
                    return false;
                }
                if (progress_) progress_->add(n);
            }
        }
        return true;
    }

    const float* floats(const GgufFile& gg, const std::string& name, int64_t elems, std::string& err, int rank = 0) {
        const TensorInfo* t = need(gg, name, kTypeF32, elems, err);
        return t ? (const float*) upload(gg, *t, err, rank) : nullptr;
    }

    // two BF16 tensors in one buffer, the first then the second (each card's share of them, back to back)
    const uint16_t* bf16_pair(const GgufFile& gg, const std::string& a, const std::string& b, int64_t elems, std::string& err,
                              int rank = 0) {
        const TensorInfo* ta = need(gg, a, kTypeBf16, elems, err);
        const TensorInfo* tb = ta ? need(gg, b, kTypeBf16, elems, err) : nullptr;
        if (!tb) return nullptr;
        const uint64_t na = card_payload(*ta, rank, err);
        if (!na) return nullptr;
        const uint64_t nb = card_payload(*tb, rank, err);
        if (!nb) return nullptr;
        void* d = take(na + nb, a, err);
        if (!d) return nullptr;
        if (!place(gg, *ta, rank, (uint8_t*) d, err)) return nullptr;
        if (!place(gg, *tb, rank, (uint8_t*) d + na, err)) return nullptr;   // the second starts where the kernels read it
        dev_bytes_ += na + nb;
        return (const uint16_t*) d;
    }

    // `<base>.weight` (Q8_0, NVFP4_PLANES or F8_E4M3, rows x cols) and, for the last two, its tensor scale `<base>.scale`
    // (F32 [1]; the F8 head's may be F32 [rows], a scale per row)
    bool linear(const GgufFile& gg, const std::string& base, int64_t rows, int64_t cols, dense::Linear& l, std::string& err,
                int rank = 0) {
        auto it = tensors_.find(base + ".weight");
        if (it == tensors_.end()) {
            err = "qwen35: the pack has no " + base + ".weight";
            return false;
        }
        const int type = (int) it->second.type;
        if (err = linear_type_error(base, it->second.type, rows, cols); !err.empty()) return false;
        const TensorInfo* t = need(gg, base + ".weight", type, rows * cols, err);
        if (!t || (int64_t) t->shape[0] != cols) {
            if (err.empty()) err = "qwen35: " + base + ".weight has the wrong shape";
            return false;
        }
        // the card's copy is a matrix of its own shape, in the pack's layout; the products read it as the unsplit one
        dense::MatrixShard s;
        if (!shard_of(*t, rank, s, err)) return false;
        l.type = type;
        l.rows = s.whole() ? rows : s.shard_rows();
        l.cols = s.whole() ? cols : s.shard_cols();
        l.w = upload(gg, *t, err, rank);
        if (!l.w) return false;
        int dev = 0;
        cudaGetDevice(&dev);
        l.dev = dev;
        if (tp_ > 0) {
            l.shard = rank;
            l.shards = tp_;
        }
        if (type == kTypeNvfp4Planes || type == kTypeF8) {
            auto sc = tensors_.find(base + ".scale");
            if (type == kTypeF8 && base == "output" && sc != tensors_.end() && (int64_t) sc->second.elements() == rows && rows > 1) {
                l.row_scales = floats(gg, base + ".scale", rows, err, rank);   // a scale per row, split with the head's rows
                return l.row_scales != nullptr;
            }
            l.scale = floats(gg, base + ".scale", 1, err, rank);   // one value every card applies to its own shard
            if (!l.scale) return false;
        }
        return true;
    }

    model::ModelDesc desc_;
    dense::ModelPtr model_;
    std::map<std::string, TensorInfo> tensors_;
    struct Block {
        uint8_t* base = nullptr;
        uint64_t size = 0, used = 0;
    };
    std::map<int, Block> blocks_;        // per CUDA device: its weights, one allocation
    size_t n_tensors_ = 0;
    uint64_t dev_bytes_ = 0;
    void* embd_ = nullptr;
    int fd_ = -1;
    std::vector<uint8_t> staging_;
    std::vector<uint8_t> whole_, cut_;   // a split tensor read whole from the pack, and the card's copy cut from it
    dense::Geometry geom_;               // the geometry the shard rules read while loading
    int split_at_ = 0;                   // the first layer on the second card (0: no split)
    int tp_ = 0;                         // the pair's width, every matrix split between them (0: not tensor parallel)
    uint64_t peer_bytes_ = 0;            // the weights there
    core::LoadProgress* progress_ = nullptr;
};

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return "qwen35"; }
    std::string validate(const GgufFile& g) const override { return validate_meta(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }

    model::Result load(const model::LoadRequest& req, std::unique_ptr<model::LoadedModel>& out) const override {
        if (req.paths.empty()) return {model::Status::Error, "qwen35: no model path"};
        auto m = std::make_unique<Loaded>();
        std::string err;
        if (!m->load(req, err)) return {model::Status::Error, err};
        out = std::move(m);
        return {};
    }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "qwen"}; }
    PreTokenizer pretokenizer() const override { return {"qwen35", program::tokenizer::pretok_qwen35}; }
    model::TokenNames token_names() const override { return program::qwen_token_names(); }
    program::grammar::ToolForm tool_form() const override { return program::qwen_tool_form(); }
    int max_expert_cards() const override { return 0; }
    bool second_card_reads_prompts() const override { return true; }
    // the load's own plan (plan_pack) on the flags and both cards' free memory before it
    bool second_card_runs_layers(const program::CommonOptions& o, double main_gbps, double peer_gbps) const override {
        if (o.diag_split_at > 0 || o.diag_tp > 0) return true;
        core::ModelFile file;   // the pack's tensor table, read in place (as the load reads it)
        std::string err;
        if (!file.open(o.model, err) || !file.load_meta(err) || !file.register_gguf_view(err)) return false;
        const GgufFile gg(o.model);
        const dense::Geometry g = geometry(gg, err);
        if (!err.empty()) return false;
        const bool draft = o.spec >= 2 && o.spec_oracle.empty();
        const dense::Options opt = run_options(gg, o.max_context, std::max(o.batch, 1), o.spec, draft, o.vision, o.kv, o.kv_resident);
        dense::BlockDrafter bd;
        if (opt.drafter == dense::Drafter::Block && !block_meta(gg, bd, g.n_embd, err)) return false;
        const dense::SplitPlan p = plan_pack(gg, g, opt, bd, 1, main_gbps, peer_gbps, o.vram_reserve_given ? o.vram_reserve_mib : 0,
                                             o.diag_split_at, o.diag_tp);
        if (o.diag_tp > 0) {
            if (!p.error.empty()) {
                basalt::log::warn("basalt qwen35", "%s", p.error.c_str());
                return false;
            }
            return true;   // the pair needs no layer cut: the roles are the pair's
        }
        return p.error.empty() && p.cut > 0;
    }

    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        (void) code;
        if (model::parse_rope_flag(a, args, o.rope)) {}
        else if (a == "--spec") { o.spec = std::atoi(args.next("--spec")); o.spec_given = true; }
        else if (a == "--kv") o.kv = args.next("--kv");
        else if (a == "--kv-resident") o.kv_resident = std::atoll(args.next("--kv-resident"));
        else if (a == "--diag-split-at") o.diag_split_at = std::atoi(args.next("--diag-split-at"));
        else if (a == "--diag-tp") o.diag_tp = std::atoi(args.next("--diag-tp"));
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(args.next("--suffix-draft")));
        else if (a == "--draft-log") basalt::core::diagnostics().draft_log = args.next("--draft-log");
        else return false;
        return true;
    }

    // the pack's drafter runs in windows of up to 16 unless --spec says otherwise (0 or 1: plain decode)
    int planned_spec(const program::CommonOptions& o) const override {
        if (o.spec_given) return o.spec >= 2 ? o.spec : 0;
        return has_drafter(o.model) ? kDefaultSpec : 0;
    }

    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        if (!o.kv.empty() && o.kv != "fp16" && o.kv != "int8") {
            basalt::log::error("basalt qwen35", "--kv %s: fp16 (the default) or int8", o.kv.c_str());
            return 2;
        }
        if (const auto code = model::check_rope_flags(o.rope, "basalt qwen35")) return code;
        if (o.kv_resident < 0) {
            basalt::log::error("basalt qwen35", "--kv-resident %lld: cells per attention layer kept in VRAM, or 0 (all)",
                               (long long) o.kv_resident);
            return 2;
        }
        if (o.spec_given && (o.spec < 0 || o.spec > 16)) {
            basalt::log::error("basalt qwen35", "--spec %d: 0 (off) to 16", o.spec);
            return 2;
        }
        if (o.diag_tp != 0 && o.diag_tp != 2) {
            basalt::log::error("basalt qwen35", "--diag-tp %d: 2 (a pair of cards), or 0 (off)", o.diag_tp);
            return 2;
        }
        if (o.diag_tp > 0 && o.diag_split_at > 0) {
            basalt::log::error("basalt qwen35", "--diag-tp and --diag-split-at are two ways to use a second card, not both");
            return 2;
        }
        if (!o.spec_given) o.spec = has_drafter(o.model) ? kDefaultSpec : 0;
        if (o.spec >= 2 && !has_drafter(o.model)) {
            basalt::log::warn("basalt qwen35", "--spec %d off: the pack has no drafter layer", o.spec);
            o.spec = 0;
        }
        return std::nullopt;
    }

private:
    static constexpr int kDefaultSpec = 16;
    static bool has_drafter(const std::string& path) {
        core::ModelFile file;
        std::string err;
        if (path.empty() || !file.open(path, err) || !file.load_meta(err)) return false;
        const core::Json* t = file.meta().get("tensors");
        return t != nullptr && (t->get("mtp.fc.weight") != nullptr || t->get("dflash.fc.0.weight") != nullptr);
    }
    const char* model_name() const override { return "Qwen3.8-27B"; }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::qwen35
