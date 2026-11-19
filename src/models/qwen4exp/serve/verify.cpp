// src/models/qwen4exp/serve/verify.cpp - verify operations for the persistent engine.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_state.hpp"
#include "../load/fn_tp_pair.hpp"   // the pair handle: qfn_runtime.hpp only forward-declares it

namespace basalt::models::qwen4exp::serve {

basalt::core::Verifier & QfnRuntime::stage_ver(int st) {
    return st == 0 ? ver : split_same ? ver_same : stages[(size_t) st - 1]->ver;
}

int QfnRuntime::pcie_num_of(double f) { return std::max(0, std::min(256, (int) (f * 256.0 + 0.5))); }

basalt::core::Verifier & QfnRuntime::stage_verb(int st) { return st == 0 ? ver_b : stages[(size_t) st - 1]->ver_b; }

int QfnRuntime::init_verify() {

    d_hist = nullptr;
    hist_stage = std::vector<int32_t>(kHistSlots, -1);
    hist_dev = last_st ? last_st->dev : -1;   // with the head: the last stage's device
    if (const basalt::core::OnDevice on_h(hist_dev); cudaMalloc(&d_hist, kHistSlots * sizeof(int32_t)) != cudaSuccess) {
        basalt::log::error("basalt serve", "the penalty-history allocation failed");
        return 1;
    }

    vh.d_res = thits.d_res;
    vh.h_res = host_res.empty() ? nullptr : host_res.data();
    vh.cache_base = thits.cache_base;
    vh.blob = thits.blob;
    vh.slot_off = xcache.slot_offsets();   // the device plan's pointers
    vh.n_slots = xcache.slots();

    pair_snap = nullptr;

    n_stages = split_devs.empty() ? 1 : (int) split_at.size() + 1;
    // two stages, the drafter and one conversation at a time: windows are pipelined over two chains of verifiers
    pipe_on = pipe_want && n_stages == 2;

    if (fn_tp != nullptr) {   // ---- TENSOR PAIR (qwen4exp --tp 2): this slice runs it solo-spec-off only.
        // The pair's windows alternate the two cards' graphs per layer and their exchanges are lockstep
        // (src/dense/forward.cpp:794-828): a batch window, a speculating window, a layer split or a pipelined
        // chain would give the cards different work, so those modes stay refused until the pair is proven.
        const char* off = o.batch > 1 ? "--batch > 1" : o.spec >= 2 ? "--spec >= 2" : pipe_want ? "--pipeline"
                          : !split_devs.empty() ? "--layer-split / a second stage"
                          : multi_gpu ? "--gpu with a peer device (the pair IS the second card)" : nullptr;
        if (off != nullptr) {
            basalt::log::error("basalt serve", "tensor pair: pair mode is solo, spec-off, single-session - %s is not paired yet", off);
            return 1;
        }
        // rank 1's mapped staging is written by rank 0's graph (the pair's hand-off wake) and read across
        // the link: the Portable flag on mapped host blocks, which qfn_run.cpp:95 sets for peer modes, is
        // needed for the pair too - set it before either Verifier allocates its staging.
        basalt::core::set_peer_portable(true);
    }

    if (n_stages > 1) {
        const size_t hb = (size_t) basalt::kernels::kVerifyMaxT *
                          (size_t) basalt::core::Verifier::handoff_floats(g) * sizeof(float);
        std::vector<float*> hand((size_t) n_stages - 1, nullptr);
        for (float*& h : hand) {
            float* hh = nullptr;
            if (cudaHostAlloc((void**) &hh, hb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &h, hh, 0) != cudaSuccess) {
                basalt::log::error("basalt serve", "the layer-split hand-off allocation failed");
                return 1;
            }
            std::memset(hh, 0, hb);
        }
        split_drive.base = &drive;
        split_drive.n = n_stages;
        for (int st = 0; st < n_stages; ++st) {
            stage_ver(st).set_route_tail_skip(o.route_tail_skip);
            stage_ver(st).set_route_prior(o.route_prior);
            stage_ver(st).set_stage(st == 0 ? 0 : split_at[(size_t) st - 1], st + 1 < n_stages ? split_at[(size_t) st] : -1,
                                    st == 0 ? nullptr : hand[(size_t) st - 1], st + 1 < n_stages ? hand[(size_t) st] : nullptr);
            split_drive.end[st] = st + 1 < n_stages ? split_at[(size_t) st] : g.n_layers;
            split_drive.cache_base[st] = drive.d.cache_base;
            split_drive.cache_slot_off[st] = drive.d.cache_slot_off;
            split_drive.pcie_num[st] = pcie_num_of(o.pcie_frac);
        }
        for (int st = 1; st < n_stages; ++st) {
            bool ok_s = false;
            if (split_same) {
                ok_s = ver_same.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err);
            } else {
                GpuStage& gs = *stages[(size_t) st - 1];
                const basalt::core::OnDevice on(gs.dev);
                basalt::core::VerifyHits vs;
                vs.d_res = gs.d_res;
                vs.h_res = host_res.empty() ? nullptr : host_res.data();
                vs.cache_base = gs.cache.device_slot(0);
                vs.blob = thits.blob;
                vs.slot_off = gs.cache.slot_offsets();
                vs.n_slots = gs.cache.slots();
                ok_s = gs.ver.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr, std::max({o.spec, o.batch, 2}), err);
                split_drive.cache_base[st] = gs.cache.device_slot(0);
                split_drive.cache_slot_off[st] = gs.cache.slot_offsets();
                split_drive.pcie_num[st] = pcie_num_of(gs.pcie_frac);
            }
            if (!ok_s) {
                basalt::log::error("basalt serve", "layer split, stage %d: %s", st + 1, err.c_str());
                return 1;
            }
        }
        for (int st = 0; st + 1 < n_stages; ++st) stage_ver(st).set_next(&stage_ver(st + 1), &split_drive);
        if (pipe_on) {
            GpuStage& gs = *stages[0];
            float* hb_host = nullptr;
            float* hand_b = nullptr;
            if (cudaHostAlloc((void**) &hb_host, hb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &hand_b, hb_host, 0) != cudaSuccess ||
                cudaMalloc(&pair_snap, basalt::core::Verifier::pair_snapshot_bytes(g, ss, 0, split_at[0])) != cudaSuccess) {
                basalt::log::error("basalt serve", "the pipelined windows' buffers failed");
                return 1;
            }
            std::memset(hb_host, 0, hb);
            ver_b.set_route_tail_skip(o.route_tail_skip);
            gs.ver_b.set_route_tail_skip(o.route_tail_skip);
            ver_b.set_route_prior(o.route_prior);
            gs.ver_b.set_route_prior(o.route_prior);
            ver_b.set_stage(0, split_at[0], nullptr, hand_b);
            gs.ver_b.set_stage(split_at[0], -1, hand_b, nullptr);
            ver.set_pipelined(pair_snap);
            ver_b.set_pipelined(pair_snap);
            const basalt::core::OnDevice on(gs.dev);
            basalt::core::VerifyHits vs;
            vs.d_res = gs.d_res;
            vs.h_res = host_res.empty() ? nullptr : host_res.data();
            vs.cache_base = gs.cache.device_slot(0);
            vs.blob = thits.blob;
            vs.slot_off = gs.cache.slot_offsets();
            vs.n_slots = gs.cache.slots();
            if (!gs.ver_b.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr, std::max({o.spec, o.batch, 2}), err)) {
                basalt::log::error("basalt serve", "pipelined windows, stage 2: %s", err.c_str());
                return 1;
            }
            ver_b.set_next(&gs.ver_b, &split_drive_b);
            split_drive_b = split_drive;
            basalt::log::info("basalt serve", "pipelined windows: on");
        }
        std::string plan_s = "0-" + std::to_string(split_at[0] - 1) + " (CUDA0)";
        for (int st = 1; st < n_stages; ++st)
            plan_s += ", " + std::to_string(split_at[(size_t) st - 1]) + "-" + std::to_string(split_drive.end[st] - 1) +
                      " (CUDA" + std::to_string(split_same ? 0 : stages[(size_t) st - 1]->dev) + ")";
        basalt::log::info("basalt serve", "layer split: layers %s, one hand-off per window", plan_s.c_str());
    }
    ver.set_route_tail_skip(o.route_tail_skip);
    ver.set_route_prior(o.route_prior);
    ver.set_main_priority(o.batch > 0);   // beside a prompt read the slots' windows must win the SMs
    if (fn_tp == nullptr && peer.valid()) ver.set_peer(&peer);   // the window drives the peer tier itself
    if (fn_tp == nullptr)
        for (const auto& xpe : xpeers)
            if (xpe->valid()) ver.add_peer(xpe.get());
    if (fn_tp != nullptr) {
        // pair-dbg (32c-3 bring-up segfault hunt): the handle was alive at hand-off; log it here, here, and
        // in dispatch so the window log says WHICH hop lost it. Remove when the segfault is fixed.
        basalt::log::error("basalt serve", "pair-dbg: at pair-block entry fn_tp=%p (&0x6F?%d) this=%p", (void*) fn_tp,
                           (int) ((uintptr_t) fn_tp == 0x6F), (void*) this);
        // the pair does not run the persistent peer window: card 1 hosts rank 1's graph instead (the dense
        // pair's law, src/dense/forward.cpp:794-828).  The tier FILL is untouched - peer_experts placement
        // still ran at load - only the WINDOW-driving set_peer/add_peer above is skipped for rank 0.
        basalt::log::info("basalt serve", "tensor pair: expert tier stays card1's, the pair's MoE exchange "
                             "lands with the next slice");
        // one mailbox for the pair (attach_tp, src/dense/stage.cpp:945-985): two slots - a window's
        // exchanges alternate between them - of 98304 B: kVerifyMaxT bf16 model-head rows (8 * 6144 * 2);
        // the f32 MoE rows (8 * 2560 * 4) fit the same slots when 32c-4 starts exchanging them.
        if (!fn_tp->open_mb(0, 2, 98304, err)) {
            basalt::log::error("basalt serve", "tensor pair: the mailbox failed to open: %s", err.c_str());
            return 1;
        }
        fn_tp_role[0] = {&fn_tp->mb, 0, fn_tp->g_full, {0, 0}, nullptr};
        fn_tp_role[1] = {&fn_tp->mb, 1, fn_tp->g_full, {0, 0}, nullptr};
        ver.set_tp(&fn_tp_role[0]);   // before init: init carves the exchange's scratch for the full rows (32c-2)
    }
    ver.set_cpu_warm(drive.d.pool, drive.d.src);
    if (fn_tp != nullptr) {
        // the pair's hand-off: rank 0 embeds the tokens on card 0 (the embedding table is a card-0 slab) and
        // every pair window PUBLISHES its start rows so rank 1's graph can begin from the same state
        // (record_inputs.cpp's stage branch, 32c-3).  One mapped block, sized like a layer split's hand-off
        // (the block above) - handoff_floats is the SAME in either geometry (hc * n_embd + n_embd + hc, and
        // the card plan halves only head widths), so g and card geometry price it identically; the FULL g
        // names it because the pair's rows are the model's rows on both cards.
        const size_t hb_tp = (size_t) basalt::kernels::kVerifyMaxT *
                             (size_t) basalt::core::Verifier::handoff_floats(*fn_tp->g_full) * sizeof(float);
        if (cudaHostAlloc((void**) &tp_hand_host, hb_tp, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &tp_hand, tp_hand_host, 0) != cudaSuccess) {
            basalt::log::error("basalt serve", "tensor pair: the hand-off allocation failed");
            return 1;
        }
        std::memset(tp_hand_host, 0, hb_tp);
        // the hand-off buffers reach init()'s stage checks (verify.cpp's set_stage docs): rank 0 runs the
        // whole model AND publishes, rank 1 runs the whole model CONSUMING - neither truncates its layers,
        // and the recorder's publish/wait branches key on the pair's rank, not the stage range.
        ver.set_stage(0, -1, nullptr, tp_hand);
        ver_tp.set_stage(0, -1, tp_hand, nullptr);
    }
    if (!ver.init(wt, fn_tp != nullptr ? *fn_tp->card0 : g,   // pair: the recorder's geometry is the CARD's (32c-3)
                  ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err) ||
        (mtp_on && !mtp.bind(wt, multi_gpu ? &mtp_head : &native_head, ver.final_R_all(), err))) {
        basalt::log::error("basalt serve", "%s", err.c_str());
        return 1;
    }
    if (pipe_on && !ver_b.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err)) {
        basalt::log::error("basalt serve", "pipelined windows: %s", err.c_str());
        return 1;
    }
    if (fn_tp != nullptr && !init_pair_rank1())   // rank 1's session, verifier and roles (32c-3)
        return 1;
    // each stage's verifier gets its stage's slot sessions
    if (o.batch > 0) {
        for (size_t k = 0; k < bslot_ss.size(); ++k) {
            std::vector<basalt::core::SessionState*> ptrs;
            for (auto& u : bslot_ss[k]) ptrs.push_back(u.get());
            basalt::core::Verifier& vk = k == 0 ? ver : stages[k - 1]->ver;
            const basalt::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
            if (!vk.init_slots(ptrs, err)) {
                basalt::log::error("basalt serve", "--batch, stage %zu: %s", k + 1, err.c_str());
                return 1;
            }
        }
    }
    for (int st = 0; st < n_stages && n_stages > 1; ++st) {
        split_drive.plan[st] = stage_ver(st).plan_sink();
        if (pipe_on) split_drive_b.plan[st] = stage_verb(st).plan_sink();
    }
    // the pool the verify windows call: with a layer split, the wrapper that routes each layer to its stage
    win_pool_fn = n_stages > 1 ? &drive_pool_split : &drive_pool_multi;
    win_pool_user = n_stages > 1 ? (void*) &split_drive : (void*) &drive;
    return 0;
}

// rank 1's window stack (32c-3): a session carved on the CARD geometry over its own arena - imitating the
// sessions phase's CUDA0 carve (load/sessions.cpp:188-221) - and a Verifier inited under OnDevice(pair dev),
// since init binds the ambient device for its streams, graphs and buffers (decode/verify/lifecycle.cpp:115).
// NO head: the head and its native tables stay rank 0's (design line "PLE + LM head + embedding: rank0-only");
// NO mtp; the PLE's card-side matrices are the twin's replicated copies (see the wiring below).
bool QfnRuntime::init_pair_rank1() {
    const basalt::core::ModelGeometry& cg = fn_tp->plan.card;
    const basalt::core::OnDevice on_tp(fn_tp->dev);
    // KV sizing: card 1's free VRAM is already reduced by its twin load (arena + native slabs were placed
    // first).  The serve carve's exact slot_cap/kv-floor accounting is the main card's (its batch/slot math
    // is refused for the pair anyway); rank 1's session takes the same formula at the card geometry -
    // max_context cells, NKV halved - guarded by a 3000 MiB headroom for its graphs, window arena and
    // staging, and the numbers are logged so a too-tight guard reads as one line, not a failed malloc.
    const uint64_t need = basalt::core::session_bytes(cg, o.max_context, K, 0, -1);
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    constexpr uint64_t kHeadroom = 3000ull << 20;
    if (need + kHeadroom > free_b) {
        basalt::log::error("basalt serve", "tensor pair: CUDA%d holds %llu MiB free of %llu; rank 1's session at "
                             "max_context %lld would take %llu MiB + %llu MiB headroom - not paired",
                     fn_tp->dev, (unsigned long long) (free_b >> 20), (unsigned long long) (total_b >> 20),
                     (long long) o.max_context, (unsigned long long) (need >> 20), (unsigned long long) (kHeadroom >> 20));
        return false;
    }
    if (cudaMalloc(&sbuf_tp, need) != cudaSuccess) {
        basalt::log::error("basalt serve", "tensor pair: CUDA%d's session allocation (%llu MiB) failed",
                     fn_tp->dev, (unsigned long long) (need >> 20));
        return false;
    }
    sbuf_tp_bytes = need;
    if (basalt::core::session_init(cg, o.max_context, K, sbuf_tp, ss_tp, 0, -1) == 0) {
        basalt::log::error("basalt serve", "tensor pair: CUDA%d's session carve failed", fn_tp->dev);
        return false;
    }
    ss_tp.ple.hist = ss_tp.ple_hist;   // the sessions phase's rule: the carve built everything but the history
    if (cudaStreamCreateWithFlags(&tp_stream, cudaStreamNonBlocking) != cudaSuccess) {
        basalt::log::error("basalt serve", "tensor pair: CUDA%d's stream creation failed", fn_tp->dev);
        return false;
    }
    // a control vector (the speed projection): its tables on this device too - rank 1's layers apply it
    // here (the layer-split stages' rule, load/sessions.cpp:114)
    if (!basalt::kernels::cvec_replicate(err)) {
        basalt::log::error("basalt serve", "tensor pair: CUDA%d: %s", fn_tp->dev, err.c_str());
        return false;
    }
    // rank 1's PLE: the ROW staging is host-side (run_begin hashes n-grams and gathers rows into the
    // mapped h_ple_, execution.cpp:101-109 and 176-192 - the table is a host mapping, no card cut), so
    // rank 1 stages IDENTICAL rows from the identical tokens its driver passes; what must live on its card
    // is the six blk.1.ple_* matrices, which the plan REPLICATES (no ple role is sliced) - the twin's arena
    // holds them whole.  Wiring imitates load/vision.cpp:62-112 on the twin's tables; anything missing
    // leaves rank 1's run not-ready - its window then carries no PLE contribution (reported, not forced).
    const basalt::core::WeightRef* wk = fn_tp->wt.find("blk.1.ple_key.weight");
    const basalt::core::WeightRef* wv = fn_tp->wt.find("blk.1.ple_value.weight");
    const basalt::core::WeightRef* wnk = fn_tp->wt.find("blk.1.ple_norm_key.weight");
    const basalt::core::WeightRef* wnq = fn_tp->wt.find("blk.1.ple_norm_query.weight");
    const basalt::core::WeightRef* wnc = fn_tp->wt.find("blk.1.ple_norm_conv.weight");
    const basalt::core::WeightRef* wc = fn_tp->wt.find("blk.1.ple_conv1d.weight");
    const bool f16 = wc != nullptr && (wc->kind == basalt::core::WeightKind::F16InF32 ||
                                       (wc->kind == basalt::core::WeightKind::Verbatim && wc->code_bits == 0)) &&
                     wc->bytes == (uint64_t) wc->elements * 2;
    if (wk != nullptr && wv != nullptr && wnk != nullptr && wnq != nullptr && wnc != nullptr && f16) {
        if (!wk->quantized())
            ss_tp.ple.w.key_bf16 = (const uint16_t*) wk->data;
        else {
            ss_tp.ple.w.key_codes = (const uint8_t*) wk->data;
            ss_tp.ple.w.key_scales = (const float*) ((const uint8_t*) wk->data + wk->codes_bytes);
            ss_tp.ple.w.key_native_data = wk->native_data;
            ss_tp.ple.w.key_native_type = wk->native_type;
            ss_tp.ple.w.key_native_q8_1 = wk->native_q8_1;
        }
        ss_tp.ple.w.value_bf16 = (const uint16_t*) wv->data;
        ss_tp.ple.w.norm_key = (const float*) wnk->data;
        ss_tp.ple.w.norm_query = (const float*) wnq->data;
        ss_tp.ple.w.norm_conv = (const float*) wnc->data;
        ss_tp.ple.w.conv1d_f16 = (const uint16_t*) wc->data;
        ss_tp.ple.consts = basalt::kernels::ple_artifact_consts();
        ss_tp.ple.table = &ple_table;
        ss_tp.ple.token = &ss_tp.ple_token;
        ss_tp.ple.prev = ss_tp.ple_prev;
    } else
        basalt::log::warn("basalt serve", "tensor pair: CUDA%d's twin does not carry every blk.1.ple_* matrix "
                             "rank 1 needs - its windows run layer 1 without the PLE until the twin load wires it");
    // the fresh state zeroes like the solo carve's (session_zero sized by the carve's own geometry)
    basalt::core::session_zero(ss_tp, cg, nullptr, tp_stream);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        basalt::log::error("basalt serve", "tensor pair: zeroing CUDA%d's session state failed", fn_tp->dev);
        return false;
    }
    // the residency view, empty-but-named: init insists a VRAM expert tier exists (lifecycle.cpp:132-138),
    // and nothing reads these while h_res is null - the all-resident scan skips it (lifecycle.cpp:362-376)
    // and dm_init builds no device-planned tail without a peer (expert_resources.cpp:111-114), so rank 1
    // gets a plain-doorbell verifier.  cache_base names its OWN arena (a live region on its card, not rank 0's);
    // the real tier view is the next slice's set_own_tier (design memo 32c-4).  d_res is never dereferenced
    // under a null h_res; it carries rank 0's word only to satisfy the existence check.
    vh_tp.d_res = thits.d_res;
    vh_tp.h_res = nullptr;
    vh_tp.cache_base = (const uint8_t*) fn_tp->arena;
    vh_tp.slot_off = nullptr;
    vh_tp.n_slots = 0;
    vh_tp.blob = thits.blob;
    ver_tp.set_route_tail_skip(o.route_tail_skip);
    ver_tp.set_route_prior(o.route_prior);
    ver_tp.set_tp(&fn_tp_role[1]);   // before init, like rank 0's (init carves the exchange scratch)
    if (!ver_tp.init(fn_tp->wt, cg, ss_tp, vh_tp, nullptr, std::max({o.spec, o.batch, 2}), err)) {
        basalt::log::error("basalt serve", "tensor pair, CUDA%d: %s", fn_tp->dev, err.c_str());
        return false;
    }
    // rank 0's window graph wakes rank 1's staged rows on this word (record_inputs.cpp's pair branch)
    fn_tp_role[0].hand_flag = ver_tp.tp_hand_flag();
    basalt::log::info("basalt serve", "tensor pair: CUDA%d's window stack is up - session %llu MiB of %llu MiB free "
                         "(before it), card geometry %lld q-heads / %lld kv / %lld value heads",
                 fn_tp->dev, (unsigned long long) (need >> 20), (unsigned long long) (free_b >> 20),
                 (long long) cg.n_head, (long long) cg.n_head_kv, (long long) cg.ssm_v_heads);
    return true;
}

int QfnRuntime::capture_windows() {

    // the pair captures the PEER card's windows first, before either card's graph exists (the pair-driver's
    // law, src/dense/forward.cpp:794-828: a captured-while-peer-runs exchange holds the spin through the
    // whole instantiate, so both cards' graphs must exist before either launches)
    if (o.serve && fn_tp != nullptr) {
        const basalt::core::OnDevice on_tp(fn_tp->dev);
        if (!ver_tp.warm_graphs(err)) {
            basalt::log::error("basalt serve", "tensor pair: capturing CUDA%d's verify windows failed: %s",
                               fn_tp->dev, err.c_str());
            return 1;
        }
    }
    // every window width's graph is captured before READY, not by the first request that needs it
    if (o.serve && stages.empty() && !ver.warm_graphs(err)) {
        basalt::log::error("basalt serve", "capturing the verify windows failed: %s", err.c_str());
        return 1;
    }
    if (fn_tp != nullptr) {
        // the pair is now built, warmed and armed; the run itself still refuses at this, the last step
        // before READY (the gate weights.cpp carried until 32c-3 moved it here - LOAD_ONLY still means
        // "this load and bring-up IS the test": exit 0, else the start fails).
        size_t free_tp = 0, total_tp = 0;
        {
            const basalt::core::OnDevice on_tp(fn_tp->dev);
            cudaMemGetInfo(&free_tp, &total_tp);
        }
        basalt::log::info("basalt serve", "tensor pair: CUDA0+CUDA%d bring-up complete (mailbox %zu slots x %zu B, "
                             "graphs warm T=1..%d); the MoE pair exchange lands in the next slice",
                     fn_tp->dev, fn_tp->mb.slots(), fn_tp->mb.slot_bytes(), std::max({o.spec, o.batch, 2}));
        // the one-line budget the next window reads (32c-3 item 7): what each card holds and what card 1 has left.
        basalt::log::info("basalt serve", "tensor pair budget: CUDA%d arena %llu MiB + native %llu MiB + rank1 "
                             "session %llu MiB + rank1 window arena %llu MiB; mailbox %zu x %zu B; CUDA%d free "
                             "%llu MiB of %llu MiB",
                     fn_tp->dev, (unsigned long long) (fn_tp->arena_bytes >> 20),
                     (unsigned long long) (fn_tp->native.weight_bytes() >> 20),
                     (unsigned long long) ((uint64_t) sbuf_tp_bytes >> 20),
                     (unsigned long long) (ver_tp.tp_arena_bytes() >> 20),
                     fn_tp->mb.slots(), fn_tp->mb.slot_bytes(),
                     fn_tp->dev, (unsigned long long) (free_tp >> 20), (unsigned long long) (total_tp >> 20));
        if (std::getenv("BASALT_FN_TP_LOAD_ONLY") != nullptr) {
            std::fflush(stdout);
            std::exit(0);   // the bring-up line above is the test's assertion; teardown would add nothing
        }
        basalt::log::error("basalt serve", "the tensor pair's driver windows are not run-able yet - the MoE pair "
                             "exchange lands in the next slice; this build brought the pair up and warmed both "
                             "cards' graphs");
        return 1;
    }
    // the solo windows of every slot session, so a lone request in any slot runs the solo path in place
    if (o.serve && slot0_is_solo && o.batch > 1) {
        const auto tw = Clock::now();
        size_t free0 = 0, total0 = 0;
        cudaMemGetInfo(&free0, &total0);
        for (int b = 1; b < o.batch; ++b) {
            make_view(rviews[(size_t) b], b);
            if (!ver.warm_session(&rviews[(size_t) b], err)) {
                basalt::log::error("basalt serve", "capturing slot %d's solo windows failed: %s", b, err.c_str());
                return 1;
            }
        }
        size_t free1 = 0;
        cudaMemGetInfo(&free1, &total0);
        basalt::log::info("basalt serve", "solo windows of %d more slot sessions captured in %.0f ms (%.1f MiB of VRAM)",
                     o.batch - 1, std::chrono::duration<double, std::milli>(Clock::now() - tw).count(),
                     (double) ((int64_t) free0 - (int64_t) free1) / 1048576.0);
    }
    S = o.spec;
    S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, S) : S;   // the MTP's windows; suffixes go up to S
    if (mtp_on && (pipe_want || S_mtp < S)) mtp.set_max_drafts((pipe_on ? basalt::core::kMtpChainMax : S_mtp) - 1);
    if (mtp_on && !mtp.prewarm(err)) {
        basalt::log::error("basalt serve", "capturing the draft graphs failed: %s", err.c_str());
        return 1;
    }
    ple_table.rearm_keepalive();   // a long load must not lapse the SSD keep-alive window before the first request
    // the tiers' placement at READY, which PROBE adapt puts back
    res_at_start = host_res;
    peer_at_start = peer.valid() ? peer.residency() : std::vector<int32_t>();
    prof_rank_start = prof_rank;
    // the elastic KV's ceiling with everything loaded; the server's copy follows it from here
    kv_ceiling_note();
    if (kv_ceiling_sent >= 0) {
        const int n = std::max(o.batch, 1);
        basalt::log::info("basalt serve", "elastic KV: about %lld cells fit in the %d session%s (%lld at full "
                          "context) before the expert cache reaches the %lld slots a prompt chunk needs",
                          (long long) kv_ceiling_sent, n, n == 1 ? "" : "s", (long long) n * o.max_context,
                          (long long) cache_floor_slots());
    }
    std::printf("INFO %s\n", basalt::program::spill_words(tier_spill).c_str());   // experts only the file holds
    std::printf("READY %lld stop\n", (long long) o.max_context);   // "stop": this engine honours STOP
    std::fflush(stdout);

    rounds = 0;
    sfx.emplace(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
    policy.emplace(S, 0.03, false);   // MTP or lookup window, learned over the whole process
    // The vision path (--vision): GENI <max_new> <embeddings file> <id,id,...> carries images and videos.  The
    // file is one or more basalt-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) in prompt
    // order, one per picture and one per two-frame video group; each record's rows go to its run of <|image_pad|> or
    // <|video_pad|> tokens, whose M-RoPE positions are Qwen3-VL's (mrope_positions.hpp): t = p, h = p + y,
    // w = p + x, and the text after the run continues at p + max(nx, ny).
    mrope_identity = true;

    // ---- VRAM <reserve_mib>, between requests, only with --vram-elastic.  It shrinks the expert cache until that
    // much VRAM is free for other programs, or grows it back towards its full size when more than that is free.
    // Never on its own: only this command (the server's POST /v1/vram) moves it.  A shrink gives back the cache's
    // last segments: the experts in their slots become CPU misses, as any expert outside the cache is (no output
    // changes beyond what a smaller cache decodes; in the resident RAM mode each is first copied to a RAM home),
    // the prompt path's loan moves down to the end of what is left
    // (a smaller chunk when it no longer fits).  A grow maps them again and puts each slot's expert back (or, when
    // the adaptive tier already brought that one back, the most-routed missing expert of the same layer); the
    // adaptive tier carries on from there.
    chunk_full = o.prefill_chunk;              // the loan's chunk with the whole cache

    // ---- the elastic KV (kKvElasticFloor): a session's K/V pools back their cells on demand, out of the expert
    // cache's tail.  `kv_fit` backs `cells` of a session before anything writes them: the work in flight lands,
    // the cache gives back its last segments (their experts become CPU misses, read from their RAM copies), and the
    // pools map that VRAM, zeroed.  `kv_release` gives a session's cells past `cells` back and the cache maps its
    // tail again, each returned slot refilled from RAM.  Both run between windows, never inside one.
    // a request's room past its prompt, and each later step (batch slots: a smaller step, as their base)
    kKvGrowCells = o.batch > 0 ? 4096 : 16384;
    kv_moves = 0;

    kv_short = false;   // the last kv_fit refused: the cells would take the cache below its floor
    kv_keep_routed = false;
    return 0;
}

}  // namespace basalt::models::qwen4exp::serve
