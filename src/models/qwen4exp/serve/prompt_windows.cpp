// src/models/qwen4exp/serve/prompt_windows.cpp - request prompt_windows operations.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"
#include "tp_window.hpp"   // the pair's window driver: pair-mode prompt parts read through it

namespace basalt::models::qwen4exp::serve {

// A short part of the prompt - the new message of a chat that continues from a checkpoint, the assistant
// header - goes through the verify windows, S tokens at a time, as decode reads them.  The batched path has
// a high fixed cost per run however few tokens it has (it streams every expert the chunk routes to that is
// not in VRAM over PCIe) and it borrows slots it must refill after; a window costs little per token, with
// the misses on the CPU.  Each part below is decided on its own, so a long first message is read batched
// and its header still goes through the windows.  Picture rows need the batched path.
bool QfnRequest::windows_ok(int64_t a, int64_t b) {
    // the tensor pair reads EVERY text part as windows: the batched prompt path (serve/prompt.cpp's sp)
    // runs rank 0's SLICED weights full-width - the pair has not wired it (per-rank chunked prefill is a
    // later slice), so the window path the pair driver owns (serve/tp_window.cpp) is the only correct one.
    // Picture rows still need the batched path; pair mode refuses --vision, so there are none.
    if (runtime_.fn_tp != nullptr) {
        if (in_slot && !solo_req) return false;
        if (runtime_.sp.embd_rows != nullptr)
            for (int64_t i = a; i < b; ++i)
                if (runtime_.sp.embd_rows[i] != nullptr) return false;
        return b > a;
    }
    if ((in_slot && !solo_req) || b - a > runtime_.o.short_read) return false;   // (the verifier is bound to this request's state)
    if (runtime_.sp.embd_rows != nullptr)
        for (int64_t i = a; i < b; ++i)
            if (runtime_.sp.embd_rows[i] != nullptr) return false;
    return true;
}

// tokens [a, b) through the windows: commit all of them, then give the draft layer their residuals
bool QfnRequest::read_windows(int64_t a, int64_t b, std::string& e) {
    basalt::core::progress_at("reading the prompt (verify windows), from token", a);   // not "batched"
    // every token is committed and the picks are discarded: no head sampling (see set_head_sampling)
    struct NoHeadSampling {
        basalt::core::Verifier& v;
        explicit NoHeadSampling(basalt::core::Verifier& x) : v(x) { v.set_head_sampling(false); }
        ~NoHeadSampling() { v.set_head_sampling(true); }
    } no_head_sampling(runtime_.ver);
    std::vector<int32_t> win((size_t) runtime_.S), outw((size_t) runtime_.S), nxt((size_t) runtime_.S);
    const bool pair = runtime_.fn_tp != nullptr;   // both cards' windows, one host (serve/tp_window.cpp)
    for (int64_t q = a; q < b;) {
        if (runtime_.halted()) { e = "cancelled"; return false; }
        const int T = (int) std::min<int64_t>(runtime_.S, b - q);
        for (int t = 0; t < T; ++t) {
            win[(size_t) t] = (int32_t) runtime_.cur[(size_t) (q + t)];
            nxt[(size_t) t] = (int32_t) runtime_.cur[(size_t) (q + t + 1)];
        }
        runtime_.drive.d.layers = 0;
        runtime_.drive.d.experts = 0;
        runtime_.drive.d.failed = false;
        const bool ran = pair
            ? run_tp_window(runtime_, T, win.data(), q, runtime_.win_pool_fn, runtime_.win_pool_user, outw, e)
            : runtime_.ver.run(T, win.data(), q, runtime_.win_pool_fn, runtime_.win_pool_user, outw.data(), e);
        if (!ran || runtime_.drive.d.failed) {
            if (runtime_.drive.d.failed && runtime_.drive.d.fail) e = runtime_.drive.d.fail;
            return false;
        }
        // parts 3: the session commit AND the PLE prev advance, exactly the solo commit(T, e) default
        const bool committed = pair ? run_tp_commit(runtime_, T, e, 3) : runtime_.ver.commit(T, e);
        if (!committed || (runtime_.mtp_on && !runtime_.mtp.prefill(runtime_.ver.final_R_all(), nxt.data(), T, q, e))) return false;
        q += T;
        runtime_.pp_reached = q;
    }
    // the batched prompt path (other streams), checkpoints and snapshots may follow: the last commit first
    if (!runtime_.ver.wait_commit(e)) return false;
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - runtime_.pp_t0).count();
    std::printf("PP %lld %lld %.0f %.1f\n", (long long) b, (long long) runtime_.pp_total, ms,
                ms > 0.0 ? 1000.0 * (double) (b - runtime_.pp_from) / ms : 0.0);
    basalt::core::progress_beat();
    std::fflush(stdout);
    return true;
}

}  // namespace basalt::models::qwen4exp::serve
