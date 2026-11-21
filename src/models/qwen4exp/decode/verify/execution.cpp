#include "common.hpp"
#include "basalt/models/qwen4exp/kernels/q80_vhead.hpp"
#include "expert_plan.hpp"
#include "diagnostics.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/models/qwen4exp/geometry.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/qwen4exp/kernels/ngram.hpp"
#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"
#include "basalt/kernels/wait_abort.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>
#include <immintrin.h>

namespace basalt::core {
using namespace verify_detail;

void Verifier::stage_inputs(int T, const int32_t* tokens, int64_t pos0) {
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    int32_t* const pk = h_pos_ + (size_t) max_t_ * g.n_head;
    int32_t* const pi = pk + (size_t) max_t_ * g.n_head_kv;
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        const int32_t pos_t = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = pos_t;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = pos_t;
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = pos_t;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) (h_flag_ + 8) = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    staged_ = true;
}

// One window through every stage: all of them launch first (a later stage's graph waits on the hand-off flag its
// predecessor's graph raises), then the host serves each stage's rings in turn.
bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
    const std::function<void()> served = std::move(served_hook_);
    served_hook_ = nullptr;
    for (Verifier* v = this; v != nullptr; v = v->next_)
        if (v->lb_ > 0 && v->h_flag_ != nullptr) *(volatile uint32_t*) (v->h_flag_ + 12) = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for (Verifier* v = this; v != nullptr; v = v->next_)
        if (!v->run_begin(T, tokens, pos0, err)) return false;
    void* u = user;
    for (Verifier* v = this; v != nullptr; v = v->next_) {
        if (!v->run_serve(pool, u, err)) return false;
        if (v->next_ == nullptr && served) served();
        if (!v->run_end(v->next_ == nullptr ? out : nullptr, err)) return false;
        u = v->next_user_;
    }
    return true;
}

// The window's staging and launch, and the flags that need no host service.  `launch=false` (the pair
// driver, serve/tp_window.cpp) stages and captures and leaves the graph unlaunched: the law is
// capture-both-then-launch (src/dense/forward.cpp:794-828), and tp_launch fires it back-to-back.
bool Verifier::run_begin(int T, const int32_t* tokens, int64_t pos0, std::string& err, cudaEvent_t after, bool launch) {
    using namespace basalt::kernels;
    const OnDevice on_device(device_);
    last_batch_ = false;
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU; restart the engine"; return false; }
    if (wait_abort_code() != 0) { err = "verify: a device wait timed out in an earlier window; restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    refresh_ar();
    if (!capture(T, err) || !capture_commit(err)) return false;
    DecodeGaps& gm = decode_gaps();
    gm.mark(11);
    const Clock::time_point t0 = Clock::now();
    decode_events().record(DecodeEvents::kRunEntry, cs_, 1);
    if (!staged_) stage_inputs(T, tokens, pos0);
    staged_ = false;
    const bool do_ple = ss.ple.ready() && ple_stage();
    fl_.ple_rows.resize((size_t) kVerifyMaxT * PLE_N_HEADS);
    uint32_t* const ple_rows = fl_.ple_rows.data();
    if (do_ple) {
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, ple_rows + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
            ss.ple.table->prefetch_rows(ple_rows + t * PLE_N_HEADS);
        }
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    gm.mark(12);
    gs_split_ = dm_ != nullptr && gs_mixl_ != nullptr;
    if (gs_split_ && !gs_live_ && !gs_move(true, err)) return false;
    if (dm_ != nullptr) {   // this window's residency, then the peer window ahead of CUDA0's
        if (!dm_begin(err, &gm)) return false;
    }
    ms_host += ms_since(t0);
    if (prof_on_) decode_gaps().verify_end = 0;   // set again only by a collected window
    const Clock::time_point tl = Clock::now();
    DecodeEvents& dev = decode_events();
    dev.stamp(5);
    if (next_ == nullptr && dev.on) { dev.next_window(); dev.record(DecodeEvents::kVerifyPre, cs_); }
    if (ar_on() && h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    // an async commit still running: the window's first node waits for it once the commit's begin kernel has run (the
    // device then knows the epoch); if that does not happen within moments, the window follows the commit on the device
    if (!finisher_on() || (commit_pending_ && !commit_begun()))
        order_after_commit(true);
    if (after != nullptr) cudaStreamWaitEvent(cs_, after, 0);   // behind another verifier's work on the same state
    if (wait_ev_ != nullptr) {
        cudaStreamWaitEvent(cs_, wait_ev_, 0);
        wait_ev_ = nullptr;
    }
    *(volatile int32_t*) h_head_from_ = head_sampling_ ? std::clamp(head_from_, 0, T) : T;
    head_from_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    set_head_keep(head_keep());
    if (!hs_launch(T, err)) return false;
    cudaError_t le = cudaSuccess;
    if (launch) le = cudaGraphLaunch(ar_off_ ? exec_nr_[T] : exec_[T], cs_);   // (pair: tp_launch fires it)
    if (next_ == nullptr) dev.record(DecodeEvents::kVerifyPost, cs_);
    ms_launch += ms_since(tl);
    if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
    lz_pending_ = false;   // the window applies the lazy layers (lazy_on) before their GDN kernels
    // a sampled or penalized request's head sampling queued behind the window now, not after a sync
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    const bool biased = sampling_.logit_bias != nullptr;
    const bool early_sample = le_ >= g.n_layers && head_sampling_ && (sampled || hist_d_ != nullptr || biased);
    // the draft probe: row t asks for the probability of the window's token t + 1
    const bool probe = draft_probe_ && head_sampling_ && sampled && T > 1 && le_ >= g.n_layers;
    // the sampled rows' kept lists; the wide path keeps none
    top_valid_ = head_sampling_ && sampled && le_ >= g.n_layers &&
                 sampler_list_len(sampling_.top_k, (int) n_vocab_) <= kSamplerNarrowMax;
    if (probe)
        for (int t = 0; t + 1 < T; ++t) h_q_[t] = tokens[t + 1];
    if (early_sample) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        if (sampled) { sp.top_ids = m_top_; sp.n_top = kTopIds; }
        if (sampler_list_len(sp.top_k, (int) n_vocab_) > kSamplerNarrowMax && sampled && !exact_head_rows(T, err))
            return false;
        sample_tokens_probe(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, probe ? m_q_ : nullptr,
                            probe ? T - 1 : 0, probe ? m_qp_ : nullptr, cs_);
    }
    (void) cudaStreamQuery(cs_);
    if (post_launch_) { auto f = std::move(post_launch_); post_launch_ = nullptr; f(); }
    gm.mark(15);
    fl_.T = T;
    fl_.pos0 = pos0;
    fl_.k = 0;
    fl_.waiting = false;
    fl_.do_ple = do_ple;
    fl_.sampled = sampled;
    fl_.biased = biased;
    fl_.early_sample = early_sample;
    fl_.probe = probe;
    if (ar_on()) {
        if (do_ple) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            ms_host += ms_since(tp);
        }
        ar_handoff();
    } else {
    // the side branch of layer 1's PLE projections waits for the rows alone
    const bool ple_early = ple_side_ && do_ple;
    if (ple_early) {
        const Clock::time_point tp = Clock::now();
        if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        *(volatile uint32_t*) (h_flag_ + 8) = 1;
        ms_host += ms_since(tp);
    }
    fl_.ple_early = ple_early;
    }
    return true;
}

// The pair driver's launch of a captured window: run_begin(T, ..., launch=false) staged and captured, and
// this fires what it skipped.  The dense law (src/dense/forward.cpp:813-826): the peer card's graph goes
// first - its exchanges may queue behind their spins while rank 0 is still working - then this card's,
// back to back, with no host work and no event between them.
bool Verifier::tp_launch(std::string& err) {
    const OnDevice on_device(device_);
    if (fl_.T == 0) { err = "verify: pair launch before its capture"; return false; }
    const cudaError_t le = cudaGraphLaunch(ar_off_ ? exec_nr_[fl_.T] : exec_[fl_.T], cs_);
    decode_events().record(DecodeEvents::kVerifyPost, cs_);
    (void) cudaStreamQuery(cs_);
    if (le != cudaSuccess) { err = std::string("verify: pair launch: ") + cudaGetErrorString(le); return false; }
    return true;
}

// The host's side of the window: each layer's ring, served as the GPU reaches it.
bool Verifier::run_serve(PoolMultiFn pool, void* user, std::string& err) {
    for (;;) {
        const int r = serve_poll(pool, user, err);
        if (r != 0) return r > 0;
        _mm_pause();
    }
}

int Verifier::serve_poll(PoolMultiFn pool, void* user, std::string& err) {
    using namespace basalt::kernels;
    if (ar_on()) return 1;   // the graph plans every layer on the device
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const int T = fl_.T;
    const bool do_ple = fl_.do_ple, ple_early = fl_.ple_early;
    const uint32_t* const ple_rows = fl_.ple_rows.data();
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int64_t steps = le_ - lb_;
    while (fl_.k < steps) {
        const int64_t k = fl_.k;
        const int64_t l = lb_ + k;
        const uint32_t want = (uint32_t) (k + 1);
        if (!fl_.waiting) {
            fl_.waiting = true;
            fl_.wait_t0 = fl_.last_flush = Clock::now();
            fl_.spins = 0;
            progress_at("verify window: waiting for the GPU to reach layer", l);
        }
        if (*seq < want) {
            if ((++fl_.spins & 1023u) != 0) return 0;
            const auto now = Clock::now();
            if (now - fl_.last_flush > std::chrono::microseconds(2000)) {
                fl_.last_flush = now;
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return -1;
                }
            }
            if (now - fl_.wait_t0 > std::chrono::seconds(20)) {
                // the caller ends the engine; no spin kernel may outlive it
                const std::string words = dm_note();   // read before the release overwrites them
                const bool drained = release_gpu_waits(5000);
                err = "verify: timed out at layer " + std::to_string(l) + words + released_note(drained) + peer_released();
                return -1;
            }
            return 0;
        }
        fl_.waiting = false;
        const Clock::time_point a = fl_.wait_t0;
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot();
        const int n = T;
        progress_at("verify window: the CPU experts of layer", l);
        if (dm_ != nullptr) {   // the ring's own rows and tiers: the pool computes the CPU's entries only
            const basalt::kernels::DmRing& R = dm_->h_ring[k];
            sink_.ext_kind = R.kind;
            if (pool != nullptr)
                pool(user, dm_->h_x + (size_t) k * dm_->x_stride, R.ids, n, ss.k, h_ymiss_, l);
            sink_.ext_kind = nullptr;
            if (tp_diag_ != nullptr) {   // BASALT_TP_DIAG: did the CPU pool fill its miss rows? (host buffer)
                double sy = 0; const int64_t yn = (int64_t) n * ss.k * g.n_embd;
                for (int64_t i = 0; i < yn; ++i) sy += (double) h_ymiss_[i] * h_ymiss_[i];
                basalt::log::raw("TPPOOL rank=%d l=%lld n_vram=%d n_peer=%d pool=%d ymiss=%.6f",
                                 tp_ != nullptr ? tp_->rank : -1, (long long) l, (int) R.n_vram, (int) R.n_peer,
                                 pool != nullptr, std::sqrt(sy));
                int k0 = 0, k1 = 0, kcpu = 0, kskip = 0, kpeer = 0;
                for (int i = 0; i < n * ss.k; ++i) {
                    const int kd = R.kind[i];
                    if (kd == 0) ++k0; else if (kd == 1) ++k1; else if (kd == -1) ++kcpu;
                    else if (kd == 8) ++kskip; else if (kd >= 2) ++kpeer;
                }
                basalt::log::raw("TPKIND rank=%d l=%lld k0=%d k1=%d cpu=%d skip=%d peer=%d",
                                 tp_ != nullptr ? tp_->rank : -1, (long long) l, k0, k1, kcpu, kskip, kpeer);
                const auto* hp = reinterpret_cast<const basalt::kernels::DmPlan*>(h_plan_);
                char kbuf[160] = {0}, dbuf[160] = {0}; int ko = 0, dd = 0;
                for (int i = 0; i < ss.k; ++i) {
                    ko += std::snprintf(kbuf + ko, sizeof(kbuf) - ko, "%d ", hp->kind[i]);
                    dd += std::snprintf(dbuf + dd, sizeof(dbuf) - dd, "%d ", hp->dst[i]);
                }
                basalt::log::raw("TPPLAN rank=%d l=%lld cnt=[%d,%d,%d,%d] kind=[%s] dst=[%s]",
                                 tp_ != nullptr ? tp_->rank : -1, (long long) l,
                                 hp->counts[0], hp->counts[1], hp->counts[2], hp->counts[3], kbuf, dbuf);
                if (n == 1 && ss.k == 10 && l < 3) {   // the ring's rows: ids, the TRUE ring tiers, the pool's row norms
                    const int rank = tp_ != nullptr ? tp_->rank : -1;
                    char ib[128], yb[160], rb[64]; int io = 0, yo = 0, ro = 0;
                    for (int i = 0; i < 10; ++i) {
                        io += std::snprintf(ib + io, sizeof(ib) - io, "%d ", R.ids[i]);
                        ro += std::snprintf(rb + ro, sizeof(rb) - ro, "%d ", R.kind[i]);
                        double s = 0;
                        for (int64_t c = 0; c < g.n_embd; ++c) {
                            const double q = h_ymiss_[(size_t) i * g.n_embd + c];
                            s += q * q;
                        }
                        yo += std::snprintf(yb + yo, sizeof(yb) - yo, "%.4f ", std::sqrt(s));
                    }
                    basalt::log::raw("TPENTS rank=%d ord=%llu l=%lld ring=%d ids=[%s] ringk=[%s] yrows=[%s]",
                                     rank, (unsigned long long) (tp_ord_ + 1), (long long) l, (int) k, ib, rb, yb);
                    if (!dm_->shadow0.empty()) {   // the two cards' residency tables, as the route read them
                        const int ne = (int) g.n_expert;
                        const int32_t* s0 = dm_->shadow0.data() + (size_t) l * ne;
                        const int32_t* s1 = dm_->shadow1.size() >= (size_t) (l + 1) * ne
                                                ? dm_->shadow1.data() + (size_t) l * ne : nullptr;
                        int both = 0, only0 = 0, only1 = 0;
                        for (int e = 0; e < ne; ++e) {
                            const bool h0 = s0[e] >= 0, h1 = s1 != nullptr && s1[e] >= 0;
                            both += h0 && h1; only0 += h0 && !h1; only1 += !h0 && h1;
                        }
                        char a0[128], a1[128]; int o0 = 0, o1 = 0;
                        for (int i = 0; i < 10 && s1 != nullptr; ++i)
                            o0 += std::snprintf(a0 + o0, sizeof(a0) - o0, "%d ", s0[R.ids[i]]),
                            o1 += std::snprintf(a1 + o1, sizeof(a1) - o1, "%d ", s1[R.ids[i]]);
                        basalt::log::raw("TPTIER rank=%d l=%lld both=%d onlyOwn=%d onlyTwin=%d own=[%s] twin=[%s]",
                                         rank, (long long) l, both, only0, only1, a0, a1);
                    }
                }
            }
        } else if (pool != nullptr)
            pool(user, h_x_, h_ids_, n, ss.k,
                 h_ymiss_, l);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        // Layer 1's pre(1, 0) copies h_ple_ -> ple_ after Layer 0's wait_flag_ge(m_flag_, 1).
        // By collecting PLE here at k == 0 (after publishing flagA/flagB for Layer 0 so the GPU can run
        // Layer 0's VRAM experts, and before raising *flag = 1), the NVMe PLE reads overlap with both
        // MtpDrafter::draft and Layer 0's attention + router + expert execution!
        if (k == 0 && do_ple && !ple_early) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return -1;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            ms_host += ms_since(tp);
        }
        *flag = want;
        if (dm_ != nullptr && dm_->warm != nullptr && l + 1 < le_) {   // the next layer's warm-up
            CpuWarm& w = *dm_->warm;
            warm_pool_->idle_stop();
            w.layer = (int) (l + 1);
            w.n_tok = std::min(n, (int) kVerifyMaxT);
            w.xq = dm_->h_xq + (size_t) k * dm_->xq_stride;
            w.res0 = dm_->shadow0.data() + (size_t) (l + 1) * g.n_expert;
            w.res1 = dm_->pdev >= 0 ? dm_->shadow1.data() + (size_t) (l + 1) * g.n_expert : nullptr;
            w.arrive.store(0, std::memory_order_relaxed);
            w.ready.store(0, std::memory_order_relaxed);
            w.blob = nullptr;
            warm_pool_->idle_start(&warm_job, &w);
        }
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
        ++fl_.k;
    }
    if (dm_ != nullptr && dm_->warm != nullptr) warm_pool_->idle_stop();
    return 1;
}

// The window's results are garbage when one of its bounded device waits timed out (wait_abort.hpp).
bool Verifier::window_intact(std::string& err) {
    const unsigned long long code = basalt::kernels::wait_abort_code();
    if (code == 0) return true;
    err = "verify: a device wait timed out (" + basalt::kernels::wait_abort_text(code) + "); the window is discarded; restart the engine";
    return false;
}

// The zero-doorbell graph plans every layer on the device and never rings h_seq_: the host raises the hand-off once,
// after the window's PLE rows are in h_ple_, and then only waits for the window's end.
void Verifier::ar_handoff() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    *(volatile uint32_t*) h_flag_ = 1;
    *(volatile uint32_t*) (h_flag_ + 8) = 1;
}

bool Verifier::ar_plan_intact(std::string& err) {
    if (!ar_on() || ar_forced_ || h_plan_err_ == nullptr || *(volatile uint32_t*) h_plan_err_ == 0) return true;
    *(volatile uint32_t*) h_plan_err_ = 0;   // the table changed under the window
    err = "verify: the zero-doorbell plan met an expert that is not in VRAM (the residency table changed during the window)";
    return false;
}

// The window's end: the GPU's finish, the profile, and on the last stage the head's picks.
bool Verifier::run_end(int32_t* out, std::string& err) {
    using namespace basalt::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    DecodeGaps& gm = decode_gaps();
    const int T = fl_.T;
    const int64_t pos0 = fl_.pos0;
    const bool sampled = fl_.sampled, biased = fl_.biased, early_sample = fl_.early_sample, probe = fl_.probe;
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    // a window the GPU never finishes (a spin kernel that never sees its flag) holds the host here; the stall watchdog
    // then releases every verifier's GPU waits (release_live_verifiers) before it ends the engine, so no spin kernel
    // outlives the process - which could leave the GPU in a lost state.  The wait itself stays a blocking sync: polling
    // with cudaStreamQuery here would cost decode time (a core calling the driver beside the expert workers).
    const cudaError_t se = cudaStreamSynchronize(cs_);
    gm.mark(-1);
    const bool dm_ok = dm_end(err);
    if (!window_intact(err) || !dm_ok) return false;
    if (hs_two_ && head_screened_ && basalt::kernels::q80p_vhead_link_failed(hs_link_)) {
        err = "verify: the peer's share of the head did not answer; the window is discarded; restart the engine";
        return false;
    }
    if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
    if (tp_diag_ != nullptr) {   // BASALT_TP_DIAG: dump this window's per-layer norms (rank -1 = solo)
        std::vector<float> nd((size_t) g.n_layers * 8);
        cudaMemcpy(nd.data(), tp_diag_, nd.size() * sizeof(float), cudaMemcpyDeviceToHost);
        const int rank = tp_ != nullptr ? tp_->rank : -1;
        for (int64_t l = 0; l < g.n_layers; ++l)
            basalt::log::raw("TPDIAG rank=%d T=%d pos=%lld l=%lld nR=%.6f nBo=%.6f nParts=%.6f nW=%.6f nBoPre=%.6f nMix=%.6f nLogits=%.6f",
                             rank, T, (long long) pos0,
                             (long long) l, std::sqrt((double) nd[8 * l]), std::sqrt((double) nd[8 * l + 1]),
                             std::sqrt((double) nd[8 * l + 2]), std::sqrt((double) nd[8 * l + 3]),
                             std::sqrt((double) nd[8 * l + 4]), std::sqrt((double) nd[8 * l + 6]),
                             std::sqrt((double) nd[8 * l + 7]));
        if (tp_diag_ent_ != nullptr && T == 1 && g.n_expert == 512) {   // the entry rows the pass wrote
            std::vector<float> ed(3 * 10);
            cudaMemcpy(ed.data(), tp_diag_ent_, ed.size() * sizeof(float), cudaMemcpyDeviceToHost);
            for (int64_t l = 0; l < 3; ++l) {
                char b[256]; int o = 0;
                for (int j = 0; j < 10; ++j)
                    o += std::snprintf(b + o, sizeof(b) - o, "%.4f ", std::sqrt((double) ed[10 * l + j]));
                basalt::log::raw("TPENT rank=%d pos=%lld l=%lld rows=[%s]", rank, (long long) pos0, (long long) l, b);
            }
        }
        if (tp_diag_plan_ != nullptr) {   // the device plans the combine reads: layers 0..2, the window's first entries
            std::vector<int32_t> pd((size_t) g.n_layers * (sizeof(basalt::kernels::DmPlan) / 4));
            cudaMemcpy(pd.data(), tp_diag_plan_, pd.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
            const int rank = tp_ != nullptr ? tp_->rank : -1;
            for (int64_t l = 0; l < 3; ++l) {
                const auto* P = reinterpret_cast<const basalt::kernels::DmPlan*>(
                    pd.data() + l * (sizeof(basalt::kernels::DmPlan) / 4));
                char kb[128], db[128], tb[128], sb[96]; int ko = 0, do_ = 0, to = 0, so = 0;
                for (int j = 0; j < 10; ++j) {
                    ko += std::snprintf(kb + ko, sizeof(kb) - ko, "%d ", P->kind[j]);
                    do_ += std::snprintf(db + do_, sizeof(db) - do_, "%d ", P->dst[j]);
                    to += std::snprintf(tb + to, sizeof(tb) - to, "%d ", P->tok[j]);
                }
                for (int j = 0; j <= 4; ++j) so += std::snprintf(sb + so, sizeof(sb) - so, "%d ", P->start[j]);
                basalt::log::raw("TPPLAN2 rank=%d pos=%lld l=%lld cnt=[%d,%d,%d,%d] kind=[%s] dst=[%s] tok=[%s] start=[%s]",
                                 rank, (long long) pos0, (long long) l, P->counts[0], P->counts[1], P->counts[2],
                                 P->counts[3], kb, db, tb, sb);
            }
            if (tp_diag_res_ != nullptr && T == 1 && g.n_expert == 512 && lb_ == 0) {
                if (tp_diag_bits_ != nullptr) {   // v6: raw fp32 bits of the router logits + mix + xn heads
                    std::vector<int32_t> bd(3 * 288);
                    std::vector<int32_t> h1(3 * 96);
                    if (tp_diag_h1_ != nullptr) cudaMemcpy(h1.data(), tp_diag_h1_, h1.size() * 4, cudaMemcpyDeviceToHost);
                    const bool gdn_f = tp_ord_ < 8 || tp_ord_ == 78;   // v15: GDN fields only on real early windows
                    cudaMemcpy(bd.data(), tp_diag_bits_, bd.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
                    for (int64_t l = 0; l < 3; ++l) {
                        char bb[1100];
                        int o = 0;
                        for (int q = 0; q < 8; ++q) {
                            float f;
                            std::memcpy(&f, &bd[l * 288 + 120 + q], 4);
                            o += std::snprintf(bb + o, sizeof(bb) - o, "%.7g ", f);
                        }
                        o += std::snprintf(bb + o, sizeof(bb) - o, "| wM=");
                        for (int q = 0; q < 4; ++q)
                            o += std::snprintf(bb + o, sizeof(bb) - o, "%08x", (unsigned) bd[l * 288 + 56 + q]);
                        o += std::snprintf(bb + o, sizeof(bb) - o, " xqM=");
                        for (int q = 0; q < 4; ++q)
                            o += std::snprintf(bb + o, sizeof(bb) - o, "%08x", (unsigned) bd[l * 288 + 60 + q]);
                        o += std::snprintf(bb + o, sizeof(bb) - o, "| rffn=");
                        for (int q = 0; q < 8; ++q) {
                            float f;
                            std::memcpy(&f, &bd[l * 288 + 40 + q], 4);
                            o += std::snprintf(bb + o, sizeof(bb) - o, "%.6g ", f);
                        }
                        o += std::snprintf(bb + o, sizeof(bb) - o, "| boA=");
                        for (int q = 0; q < 8; ++q) {
                            float f;
                            std::memcpy(&f, &bd[l * 288 + 48 + q], 4);
                            o += std::snprintf(bb + o, sizeof(bb) - o, "%.6g ", f);
                        }
                        o += std::snprintf(bb + o, sizeof(bb) - o, "| wRt=%08x%08x%08x%08x",
                                           (unsigned) bd[l * 288 + 56], (unsigned) bd[l * 288 + 57],
                                           (unsigned) bd[l * 288 + 58], (unsigned) bd[l * 288 + 59]);
                        if (gdn_f)
                            o += std::snprintf(bb + o, sizeof(bb) - o, "| mxA=%08x%08x%08x%08x| qkv=%08x%08x%08x%08x zH=%08x%08x%08x%08x mA=%08x%08x%08x%08x mB=%08x%08x%08x%08x wout=%08x%08x%08x%08x xq=%08x%08x%08x%08x%08x%08x%08x%08x y16=%08x%08x%08x%08x y16t=%08x%08x%08x%08x y168=%08x%08x%08x%08x y1623=%08x%08x%08x%08x",
                            (unsigned) bd[l * 288 + 60], (unsigned) bd[l * 288 + 61], (unsigned) bd[l * 288 + 62], (unsigned) bd[l * 288 + 63],
                            (unsigned) bd[l * 288 + 64], (unsigned) bd[l * 288 + 65], (unsigned) bd[l * 288 + 66], (unsigned) bd[l * 288 + 67],
                            (unsigned) bd[l * 288 + 72], (unsigned) bd[l * 288 + 73], (unsigned) bd[l * 288 + 74], (unsigned) bd[l * 288 + 75],
                            (unsigned) bd[l * 288 + 80], (unsigned) bd[l * 288 + 81], (unsigned) bd[l * 288 + 82], (unsigned) bd[l * 288 + 83],
                            (unsigned) bd[l * 288 + 88], (unsigned) bd[l * 288 + 89], (unsigned) bd[l * 288 + 90], (unsigned) bd[l * 288 + 91],
                            (unsigned) bd[l * 288 + 96], (unsigned) bd[l * 288 + 97], (unsigned) bd[l * 288 + 98], (unsigned) bd[l * 288 + 99],
                            (unsigned) bd[l * 288 + 104], (unsigned) bd[l * 288 + 105], (unsigned) bd[l * 288 + 106], (unsigned) bd[l * 288 + 107],
                            (unsigned) bd[l * 288 + 108], (unsigned) bd[l * 288 + 109], (unsigned) bd[l * 288 + 110], (unsigned) bd[l * 288 + 111],
                            (unsigned) bd[l * 288 + 112], (unsigned) bd[l * 288 + 113], (unsigned) bd[l * 288 + 114], (unsigned) bd[l * 288 + 115],
                            (unsigned) bd[l * 288 + 116], (unsigned) bd[l * 288 + 117], (unsigned) bd[l * 288 + 118], (unsigned) bd[l * 288 + 119],
                            (unsigned) bd[l * 288 + 120], (unsigned) bd[l * 288 + 121], (unsigned) bd[l * 288 + 122], (unsigned) bd[l * 288 + 123],
                            (unsigned) bd[l * 288 + 124], (unsigned) bd[l * 288 + 125], (unsigned) bd[l * 288 + 126], (unsigned) bd[l * 288 + 127]);
                        basalt::log::raw("TPBITS rank=%d ord=%llu pos=%lld l=%lld bo8=[%s]", rank,
                                         (unsigned long long) (tp_ord_ + 1), (long long) pos0, (long long) l, bb);
                        {
                            char xb[1300]; int xo = 0;
                            xo += std::snprintf(xb + xo, sizeof(xb) - xo, "TPXQ rank=%d ord=%llu pos=%lld l=%lld xqr=", rank,
                                                (unsigned long long) (tp_ord_ + 1), (long long) pos0, (long long) l);
                            for (int q = 0; q < 144; ++q) xo += std::snprintf(xb + xo, sizeof(xb) - xo, "%08x", (unsigned) bd[l * 288 + 128 + q]);
                            basalt::log::raw("%s", xb);
                        }
                        {
                            char hb[900]; int ho = 0;
                            ho += std::snprintf(hb + ho, sizeof(hb) - ho, "xq=");
                            for (int q = 0; q < 48; ++q) ho += std::snprintf(hb + ho, sizeof(hb) - ho, "%08x", (unsigned) h1[l * 96 + q]);
                            ho += std::snprintf(hb + ho, sizeof(hb) - ho, " wt=");
                            for (int q = 0; q < 40; ++q) ho += std::snprintf(hb + ho, sizeof(hb) - ho, "%08x", (unsigned) h1[l * 96 + 48 + q]);
                            ho += std::snprintf(hb + ho, sizeof(hb) - ho, " ptr=%08x%08x", (unsigned) h1[l * 96 + 88], (unsigned) h1[l * 96 + 89]);
                            basalt::log::raw("TPHASH rank=%d ord=%llu pos=%lld l=%lld %s", rank,
                                             (unsigned long long) (tp_ord_ + 1), (long long) pos0, (long long) l, hb);
                        }
                    }
                    {   // HARNESS SELF-CHECK: the raw-byte slots must still hold exactly the bytes the probe
                        // hashed.  Runs 20-23 each chased a phantom capture and drew conclusions from data that
                        // could not be validated; the cause was a writer left on the old 128-slot stride, which
                        // landed layer 1's router logits INSIDE layer 0's captured head rows.  A unit the probe
                        // zeroed (the gs serve's heads past its own row) has no bytes to compare: skipped.
                        static const int kSelfHeads[4] = {0, 8, 24, 32};
                        auto fnv_unit = [](const int32_t* w) {
                            int32_t h = 0;
                            for (int i = 0; i < 36; ++i) h = h * 16777619 + w[i];   // diag_hash_u8 over 144 B
                            return h;
                        };
                        int self_ok = 0, self_tot = 0;
                        for (int64_t l = 0; l < 3; ++l)
                            for (int q = 0; q < 4; ++q) {
                                const int32_t want = h1[l * 96 + kSelfHeads[q]];
                                if (want == 0) continue;
                                ++self_tot;
                                if (fnv_unit(bd.data() + l * 288 + 128 + 36 * q) == want) ++self_ok;
                            }
                        basalt::log::raw("TPSELF rank=%d ord=%llu ok=%d/%d", rank,
                                         (unsigned long long) (tp_ord_ + 1), self_ok, self_tot);
                    }
                }
                ++tp_ord_;
            }
        }
    }
    if (!ar_plan_intact(err)) return false;
    constexpr int64_t wstat_every = 200;   // windows per report
    if (dm_ != nullptr && dm_->wstat != nullptr && (dm_->wstat_t += T, ++dm_->wstat_windows == wstat_every)) {
        // --wait-stats: the last wstat_every windows' counters
        std::vector<unsigned long long> w((size_t) kDmStatWords);
        cudaMemcpy(w.data(), dm_->wstat, w.size() * 8, cudaMemcpyDeviceToHost);
        cudaMemset(dm_->wstat, 0, w.size() * 8);
        const double nw = (double) dm_->wstat_windows;
        basalt::log::raw("basalt verify: dm_wait over %lld windows (avg T %.2f):\n%s", (long long) dm_->wstat_windows,
                     dm_->wstat_t / nw, basalt::kernels::dec_moe_stats_text(w.data(), nw).c_str());
        dm_->wstat_windows = 0;
        dm_->wstat_t = 0;
    }
    // the window does not retire the commit: whoever reads the session from the host or another stream calls wait_commit(),
    // and a finished event is noticed here at no cost
    if (commit_pending_ && cudaEventQuery(commit_done_) == cudaSuccess) commit_pending_ = false;
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    gm.mark(0);
    if (prof_on_) collect_profile();   // the window's GPU stage stamps
    gm.mark(1);
    // ---- a sampled or penalized request: the head's sampling again, host-side so its parameters are this call's
    // own (a captured kernel would replay the same draws forever).  Row t's draw is Philox(seed, pos0 + t): tied to
    // the position it samples, not to how the text was cut into windows, so a seed replays the same text whatever
    // the drafts were. Exact: a rejected row's draw is discarded, and no kept decision depends on a reused draw.
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)
        ++windows;
        return true;
    }
    if (head_sampling_ && (sampled || hist_d_ != nullptr || biased) && !early_sample) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        if (sampled) { sp.top_ids = m_top_; sp.n_top = kTopIds; }
        if (sampler_list_len(sp.top_k, (int) n_vocab_) > kSamplerNarrowMax && sampled && !exact_head_rows(T, err))
            return false;
        sample_tokens_probe(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, probe ? m_q_ : nullptr,
                            probe ? T - 1 : 0, probe ? m_qp_ : nullptr, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {   // m_out_ is the mapped h_out_: synced, it is readable
            err = "verify: the head sampling failed";
            return false;
        }
    }
    gm.mark(2);
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    gm.mark(3);
    ++windows;
    progress_at("decode");
    progress_beat();
    return true;
}

void Verifier::set_plan_slot() {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_;
    const int64_t i32 = 4 + (cap + 1) + cap + cap;
    const int64_t ptr_off = (i32 + 1) & ~1ll;
    sink_.counts = base;
    sink_.start = base + 4;
    sink_.dst = sink_.start + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr = (unsigned long long*) (base + ptr_off);
    sink_.ptr2 = sink_.ptr + cap;
    sink_.start2 = base + ptr_off + 4 * cap;
    sink_.staging_cap = kStagingBlobs;
}

// Flag B only rises.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    std::atomic_ref<uint32_t> f(*flag);
    uint32_t cur = f.load();
    while (cur < value && !f.compare_exchange_strong(cur, value)) {}
}

// The PCIe groups are planned: raise flag B for the layer (the graph waits for it before staging them).
void Verifier::raise_flag_b(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    raise_flag(v->h_flagB_, v->cur_layer_ + 1);
}

void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    _mm_sfence();
    *(volatile uint32_t*) v->h_flagA_ = v->cur_layer_ + 1;
}

}  // namespace basalt::core
