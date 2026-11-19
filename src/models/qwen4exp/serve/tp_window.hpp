// src/models/qwen4exp/serve/tp_window.hpp - the Flash-Next tensor pair's window and commit: the dense pair
// driver's LAW, applied to the two cards' verify windows (src/dense/forward.cpp forward_window, :794-828):
// per window zero both roles' exchange counters, check the mailbox's sticky hang word, capture BOTH cards'
// graphs (peer card first, launch=false), assert the cards queued equal exchange counts, then launch peer
// first and this card back to back - the streams are never joined by events, the mailbox's per-round waits
// are the pair's only true dependencies.
//
// Structurally complete, not yet exercised: run_tp_window refuses unless `pair_runnable`, and the pair's
// bring-up itself refuses the start (serve/verify.cpp's capture_windows) until the MoE pair exchange lands
// (32c-4) - so no request can reach this file's code in this slice.  The refusal is the guard; the
// choreography is what the next window must not have to re-derive.
#pragma once

#include "qfn_state.hpp"

namespace basalt::models::qwen4exp::serve {

/// One decode window on the pair: `tokens[0..T)` at positions p.., both cards' graphs captured and launched
/// under the law above; one host serves both cards' rings (rank 0's gets the CPU expert pool `pool`/`user`
/// - rank 1's poll runs with a null pool, "rank 1 never waits on the host"); `outv` takes rank 0's picks.
bool run_tp_window(QfnRuntime& rt, int T, const int32_t* tokens, int64_t p, basalt::core::PoolMultiFn pool,
                   void* user, std::vector<int32_t>& outv, std::string& err);

/// The pair's commit: both cards stage (commit launch=false), rank 0's commit graph launches, then rank 1's
/// under OnDevice(pair dev); both streams are waited before returning (no async-commit bookkeeping in pair
/// mode until the pair serves, 32c-5).  With `pair_runnable` still false it is the plain rank-0 commit.
bool run_tp_commit(QfnRuntime& rt, int n_keep, std::string& err, int parts);

}  // namespace basalt::models::qwen4exp::serve
