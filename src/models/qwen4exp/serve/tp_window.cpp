// src/models/qwen4exp/serve/tp_window.cpp - see tp_window.hpp.  The choreography below mirrors
// src/dense/forward.cpp forward_window(:794-828) line for line; the phases are the opened-up run() of
// decode/verify/execution.cpp (tp_capture/tp_launch/tp_serve_once/tp_end) and the launch-flagged commit of
// decode/verify/commit.cpp (commit(..., launch), tp_launch_commit/tp_wait_commit).
#include "tp_window.hpp"

#include "../load/fn_tp_pair.hpp"   // the pair handle (qfn_runtime.hpp only forward-declares it)
#include "basalt/core/on_device.hpp"

#include <immintrin.h>   // _mm_pause between the two cards' non-blocking ring polls

namespace basalt::models::qwen4exp::serve {

bool run_tp_window(QfnRuntime& rt, int T, const int32_t* tokens, int64_t p, basalt::core::PoolMultiFn pool,
                   void* user, std::vector<int32_t>& outv, std::string& err) {
    if (rt.fn_tp == nullptr || !rt.pair_runnable) {
        // defense in depth: pair_runnable comes up only with both cards' windows warmed and the MoE
        // partial-sum exchange in their graphs (32c-4, serve/verify.cpp's capture_windows) - a pair that
        // did not bring BOTH cards up never reaches this driver.
        err = "fn tp: the pair's driver windows are not run-able yet";
        return false;
    }
    basalt::core::TpMailbox& mb = rt.fn_tp->mb;
    if (mb.hang_reported()) {   // sticky: a mailbox round already timed out - the cards' results have split
        err = "fn tp: an earlier exchange diverged; the pair's results have split";
        return false;
    }
    rt.fn_tp_role[0].epi[0] = rt.fn_tp_role[0].epi[1] = 0;   // the window's exchange sequence starts here
    rt.fn_tp_role[1].epi[0] = rt.fn_tp_role[1].epi[1] = 0;   // (forward_window:796 - both counters, every window)
    rt.ver_tp.tp_reset_hand();   // rank 1's hand-off wake, reset like run() does for chain stages (execution.cpp:65)
    // capture BOTH cards' graphs before either launches - rank 1 first, under its own device (the
    // instantiate of a graph whose exchanges spin must not start while its peer's graph is already running).
    {
        const basalt::core::OnDevice on(rt.fn_tp->dev);
        if (!rt.ver_tp.tp_capture(T, tokens, p, err)) return false;
    }
    if (!rt.ver.tp_capture(T, tokens, p, err)) return false;
    if (rt.fn_tp_role[0].epi[0] != rt.fn_tp_role[1].epi[1])
        basalt::log::error("fn tp", "window T=%d pos=%lld: cards queued %d and %d exchanges (not lockstep)", T,
                           (long long) p, (int) rt.fn_tp_role[1].epi[1], (int) rt.fn_tp_role[0].epi[0]);
    // both cards' graphs exist and both hosts finished queueing: launch back to back, the pair's first
    // (its graph's exchanges wait for rank 0's flags, so it may queue behind them freely - forward_window:813-822)
    {
        const basalt::core::OnDevice on(rt.fn_tp->dev);
        if (!rt.ver_tp.tp_launch(err)) return false;
    }
    if (!rt.ver.tp_launch(err)) return false;
    // ONE host, both cards' rings: each poll is non-blocking (serve_poll returns 0 while the next doorbell
    // is in flight), so neither card's spin waits on the host's ordering.  Rank 0's poll drives the CPU
    // expert pool - rank 0's CPU share of the split experts, folded into its partial like any other tier
    // (32c-4: the route's kDmSkip left rank 0 exactly the experts card 1 does not hold); rank 1's runs
    // with no pool at all (its rings still need their plans published).
    bool done0 = false, done1 = false;
    std::vector<int32_t> out1((size_t) T, 0);   // rank 1's picks: read for its head to have written, then dropped
    while (!done0 || !done1) {
        if (!done0) {
            const int r = rt.ver.tp_serve_once(pool, user, err);
            if (r != 0) {
                if (r < 0) return false;
                done0 = true;
            }
        }
        if (!done1) {
            const int r = rt.ver_tp.tp_serve_once(nullptr, nullptr, err);
            if (r != 0) {
                if (r < 0) return false;
                done1 = true;
            }
        }
        if (!done0 || !done1) _mm_pause();
    }
    // both windows finish (rank 1 first: its run_end syncs its stream; its mailbox rounds already met rank
    // 0's inside the graphs); the picks read from rank 0 alone - the head and its table stay rank 0's card.
    if (!rt.ver_tp.tp_end(out1.data(), err)) return false;
    outv.resize((size_t) T);
    if (!rt.ver.tp_end(outv.data(), err)) return false;
    return true;
}

bool run_tp_commit(QfnRuntime& rt, int n_keep, std::string& err, int parts) {
    if (!rt.pair_runnable) return rt.ver.commit(n_keep, err, parts);   // (guarded like the window; solo commit)
    // both cards stage their epoch (their own last_t_/last_pos0_ - the same window, so the same commit);
    // rank 0's commit graph goes first, rank 1's follows on its own stream, then both are waited.  Rank 1's
    // commit advances its own session (its own ple_prev, its own GDN/QSA rows) exactly as rank 0's does.
    if (!rt.ver.commit(n_keep, err, parts, false)) return false;
    {
        const basalt::core::OnDevice on(rt.fn_tp->dev);
        if (!rt.ver_tp.commit(n_keep, err, parts, false)) return false;
    }
    if (!rt.ver.tp_launch_commit(err)) return false;
    {
        const basalt::core::OnDevice on(rt.fn_tp->dev);
        if (!rt.ver_tp.tp_launch_commit(err)) return false;
    }
    if (!rt.ver.tp_wait_commit(err)) return false;
    const basalt::core::OnDevice on(rt.fn_tp->dev);
    return rt.ver_tp.tp_wait_commit(err);
}

}  // namespace basalt::models::qwen4exp::serve
