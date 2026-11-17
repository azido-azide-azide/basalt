// tests/dense/tp_exchange_test.cpp - the tensor-parallel pair's all-reduce of a window's row blocks across two cards.
//
// What it has to prove, in order: that each card ends up with the SUM of the two cards' partials of the same rows (at the
// rounding the kernel promises: added as floats, rounded to BF16 once), that this holds for every exchange of a window and
// not just the first (the peer's block of exchange e must be the one the card reads, and its own write of exchange e must
// not land on a block the peer has not read yet), that the same sequence works INSIDE a graph captured on each card and
// replayed - which is how a window runs it - and what one exchange costs on this pair.
//
// The partials are distinct per exchange so a card's result is the sum of two known values, not an accumulation: each
// exchange reads its own block of the card's buffer, the way every exchange's product is its own output in the forward.
// The other form - the sum landing added into the running residual rows - is the pair's other half of the proof: both
// cards must end with rows identical to the bit, because the residual is the stream the pair keeps replicated.
// The flags are lowered between windows only after both cards' streams are idle, which is the discipline `TpHand` states.
// The replays ALTERNATE two sets of partials, so a card that answered from bytes the memory system never delivered shows
// up as a wrong sum instead of as a suspiciously fast window - the failure `copy_to_mapped_wt` exists to close.
//
// GPU, synthetic, no model.  Needs two cards with some free memory; skips (77) without them.
#include "internal.hpp"

#include "basalt/core/on_device.hpp"
#include "basalt/kernels/bf16_bits.hpp"
#include "basalt/kernels/elementwise.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace basalt;
using basalt::dense::TpHand;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

std::string cuda_err(const char* what, cudaError_t e) {
    return std::string(what) + ": " + cudaGetErrorString(e);
}

constexpr int kEmb = 5120;    // a dense model's row width, so the sizes are the ones the pair will run
constexpr int kExch = 8;      // exchanges per window here (a real window has two per layer)
constexpr int kRows = 16;     // a full decode window; the narrow case runs one row
constexpr int kNarrow = 1;

// The values the two cards start each exchange with, and the sum the exchange owes them, in TWO sets: set 1 shifts every
// value by 8.  The replay loop below alternates them, and that is the point - a replay that reads the bytes it read last
// time cannot tell a fresh answer from one the memory system never delivered, and a hand-off that answers from stale bytes
// fails in exactly that direction: measured on a two-card pair, the stale answer came back with WRONG DATA, THE RIGHT SUMS
// FOR THE OTHER SET'S NEIGHBOURS, and a FASTER window than the honest one.  Alternating sets turns that into a mismatch.
uint16_t part_a(int set, int e, int64_t i) {
    return kernels::bf16_from_f32((set ? 8.f : 0.f) + 0.5f + (float) (i % 13) + 0.25f * (float) e);
}
uint16_t part_b(int set, int e, int64_t i) {
    return kernels::bf16_from_f32((set ? 8.f : 0.f) - 0.125f + (float) ((i * 7) % 11) - 0.5f * (float) e);
}
uint16_t want(int set, int e, int64_t i) {
    return kernels::bf16_from_f32(kernels::f32_from_bf16(part_a(set, e, i)) + kernels::f32_from_bf16(part_b(set, e, i)));
}

// one card's buffer: `exchanges` fresh partials, one per exchange, at the offsets the exchanges read
bool fill(int dev, uint16_t* x, const std::vector<uint16_t>& host, int rows) {
    const core::OnDevice on(dev);
    const int64_t block = (int64_t) rows * kEmb;
    for (int e = 0; e < kExch; ++e) {
        const uint16_t* src = host.data() + (int64_t) e * block;
        if (cudaMemcpyAsync(x + (int64_t) e * block, src, (size_t) block * 2, cudaMemcpyHostToDevice) != cudaSuccess)
            return false;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

std::vector<uint16_t> host_partials(int rank, int rows, int set) {
    std::vector<uint16_t> h((size_t) kExch * rows * kEmb);
    for (int e = 0; e < kExch; ++e)
        for (int64_t i = 0; i < (int64_t) rows * kEmb; ++i)
            h[(size_t) e * rows * kEmb + i] = rank == 0 ? part_a(set, e, i) : part_b(set, e, i);
    return h;
}

// every exchange of one window, as card `rank`
void run_window(const TpHand& h, int rank, cudaStream_t s, uint16_t* x, int rows) {
    const int64_t block = (int64_t) rows * kEmb;
    for (int e = 0; e < kExch; ++e) dense::tp_allreduce2(h, rank, e, x + (int64_t) e * block, rows, s);
}

bool cuda_ok(const std::string& what, cudaError_t e) {
    if (e == cudaSuccess) return true;
    check(false, cuda_err(what.c_str(), e));
    return false;
}

// both streams drained, or the flags poisoned so a wait that can never be satisfied lets the card go: the test must never
// leave a spinning kernel on a card somebody else has to use.
bool drain(cudaStream_t s0, cudaStream_t s1, TpHand& h, const std::string& at) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const cudaError_t a = cudaStreamQuery(s0), b = cudaStreamQuery(s1);
        if (a != cudaErrorNotReady) return cuda_ok(at + " (card 0)", a);
        if (b != cudaErrorNotReady) return cuda_ok(at + " (card 1)", b);
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(20)) {
            volatile uint32_t* f = (volatile uint32_t*) h.mem;
            f[0] = 0xFFFFFFFFu;
            f[1] = 0xFFFFFFFFu;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            std::printf("FAIL %s: the pair was still waiting after 20 s; its flags were raised to let the cards go\n", at.c_str());
            ++failures;
            (void) cudaStreamSynchronize(s0);
            (void) cudaStreamSynchronize(s1);
            return false;
        }
        std::this_thread::yield();
    }
}

// the sums the pair owes, on both cards, for every exchange
bool check_sums(int dev0, int dev1, const uint16_t* x0, const uint16_t* x1, int rows, const std::string& at, int set = 0,
                bool quiet = false) {
    const int64_t block = (int64_t) rows * kEmb;
    std::vector<uint16_t> got0((size_t) kExch * block), got1((size_t) kExch * block);
    bool read_ok = true;
    {
        const core::OnDevice on(dev0);
        read_ok = read_ok && cuda_ok(at + " (read card 0)", cudaMemcpy(got0.data(), x0, got0.size() * 2, cudaMemcpyDeviceToHost));
    }
    {
        const core::OnDevice on(dev1);
        read_ok = read_ok && cuda_ok(at + " (read card 1)", cudaMemcpy(got1.data(), x1, got1.size() * 2, cudaMemcpyDeviceToHost));
    }
    if (!read_ok) return false;
    int wrong0 = 0, wrong1 = 0;
    int64_t first0 = -1, first1 = -1;
    for (int e = 0; e < kExch; ++e)
        for (int64_t i = 0; i < block; ++i) {
            const uint16_t w = want(set, e, i);
            if (got0[(size_t) e * block + i] != w) { ++wrong0; if (first0 < 0) first0 = ((int64_t) e << 40) | i; }
            if (got1[(size_t) e * block + i] != w) { ++wrong1; if (first1 < 0) first1 = ((int64_t) e << 40) | i; }
        }
    if (!quiet) {
        if (wrong0 || wrong1) {   // the first difference on each card, spelled out
            const int e0 = (int) (first0 >> 40);
            const int64_t i0 = first0 & ((1ll << 40) - 1);
            const int e1 = (int) (first1 >> 40);
            const int64_t i1 = first1 & ((1ll << 40) - 1);
            std::printf("#   card 0: %d wrong (first exchange %d value %lld: %.6f, wants %.6f)\n", wrong0, e0,
                        (long long) i0, wrong0 ? kernels::f32_from_bf16(got0[(size_t) e0 * block + i0]) : 0.0f,
                        kernels::f32_from_bf16(want(set, e0, i0)));
            std::printf("#   card 1: %d wrong (first exchange %d value %lld: %.6f, wants %.6f)\n", wrong1, e1,
                        (long long) i1, wrong1 ? kernels::f32_from_bf16(got1[(size_t) e1 * block + i1]) : 0.0f,
                        kernels::f32_from_bf16(want(set, e1, i1)));
        }
        check(wrong0 == 0, at + ": card 0's every exchange holds the two cards' sum");
        check(wrong1 == 0, at + ": card 1's every exchange holds the two cards' sum");
    }
    return wrong0 == 0 && wrong1 == 0;
}

struct Cfg {
    int dev0 = 0, dev1 = 1;
    TpHand* h = nullptr;
    uint16_t *x0 = nullptr, *x1 = nullptr;
    cudaStream_t s0 = nullptr, s1 = nullptr;
    std::vector<uint16_t> hp0, hp1, hp0b, hp1b;   // sets 0 and 1, the replays alternate them
    int rows = kRows;

    bool reset(int set = 0) {   // the exchanges' partials again, both cards idle
        return fill(dev0, x0, set ? hp0b : hp0, rows) && fill(dev1, x1, set ? hp1b : hp1, rows);
    }
};

bool run_case(const char* what, Cfg& c, int rows) {
    c.rows = rows;
    c.hp0 = host_partials(0, rows, 0);
    c.hp1 = host_partials(1, rows, 0);
    c.hp0b = host_partials(0, rows, 1);
    c.hp1b = host_partials(1, rows, 1);
    if (!c.reset()) return check(false, std::string(what) + ": the partials did not reach the cards"), false;
    // as launched
    run_window(*c.h, 0, c.s0, c.x0, rows);
    run_window(*c.h, 1, c.s1, c.x1, rows);
    if (!drain(c.s0, c.s1, *c.h, std::string(what) + ": window")) return false;
    check_sums(c.dev0, c.dev1, c.x0, c.x1, rows, std::string(what) + ": launched");
    // Both cards are past this window, so its flags go down before the next one runs - replayed or launched.  Lowering is
    // only legal here: a raise still in flight from a window that has not finished would otherwise release the next one's
    // wait before its peer's write.
    dense::tp_hand_lower(*c.h);
    // as captured per card and replayed, the flags lowered between the windows
    cudaGraphExec_t g0 = nullptr, g1 = nullptr;
    const auto capture = [&](int dev, cudaStream_t s, uint16_t* x, cudaGraphExec_t& g) {
        const core::OnDevice on(dev);
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed) != cudaSuccess) return false;
        run_window(*c.h, dev == c.dev0 ? 0 : 1, s, x, rows);
        cudaGraph_t graph = nullptr;
        if (cudaStreamEndCapture(s, &graph) != cudaSuccess) return false;
        const bool ok = cudaGraphInstantiate(&g, graph, 0) == cudaSuccess;
        cudaGraphDestroy(graph);
        return ok;
    };
    if (!capture(c.dev0, c.s0, c.x0, g0) || !capture(c.dev1, c.s1, c.x1, g1)) {
        check(false, std::string(what) + ": the pair's window did not capture");
        return false;
    }
    check(true, std::string(what) + ": each card's window captured with its exchanges inside");
    constexpr int kReplays = 30;
    const auto t0 = std::chrono::steady_clock::now();
    int bad = 0;
    // The timed loop stays clean: a readback and a host comparison inside it would be measured as the pair's exchange
    // (measured: 30.5 -> 469 us per exchange for two 160 KB readbacks and one pass over the rows).
    for (int r = 0; r < kReplays; ++r) {
        if (!c.reset(r % 2)) return check(false, std::string(what) + ": a replay's partials did not reach the cards"), false;
        if (cudaGraphLaunch(g1, c.s1) != cudaSuccess || cudaGraphLaunch(g0, c.s0) != cudaSuccess) {
            check(false, std::string(what) + ": a replay did not launch");
            return false;
        }
        if (!drain(c.s0, c.s1, *c.h, std::string(what) + ": replay " + std::to_string(r))) return false;
        dense::tp_hand_lower(*c.h);
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    check(bad == 0, std::string(what) + ": every replay of the captured window gave the sums it owes, on alternating partials");
    std::printf("# %s: %d replays of %d exchanges in %d rows: %.1f us per window, %.2f us per exchange\n", what, kReplays, kExch,
                rows, us / kReplays, us / (kReplays * (double) kExch));
    // the same captured windows again, untimed, verified one by one on alternating partials: the replay above cannot see a
    // stale answer, this one can (see the note on part_a/part_b)
    int stale = 0;
    constexpr int kChecks = 12;
    for (int r = 0; r < kChecks; ++r) {
        const int set = r % 2;
        if (!c.reset(set)) return check(false, std::string(what) + ": a checked replay's partials did not reach the cards"), false;
        if (cudaGraphLaunch(g1, c.s1) != cudaSuccess || cudaGraphLaunch(g0, c.s0) != cudaSuccess) {
            check(false, std::string(what) + ": a checked replay did not launch");
            return false;
        }
        if (!drain(c.s0, c.s1, *c.h, std::string(what) + ": checked replay " + std::to_string(r))) return false;
        dense::tp_hand_lower(*c.h);
        if (!check_sums(c.dev0, c.dev1, c.x0, c.x1, rows, std::string(what) + ": checked replay " + std::to_string(r), set,
                        /*quiet=*/r >= 3)) ++stale;
    }
    bad += stale;
    std::printf("# %s: %d checked replays on alternating partials, %d wrong\n", what, kChecks, stale);
    cudaGraphExecDestroy(g0);
    cudaGraphExecDestroy(g1);
    return bad == 0;
}

// the residual rows the window starts with: the SAME rows on both cards, which is what the pair's exchanges keep them
uint16_t resid(int64_t i) { return kernels::bf16_from_f32(1.25f + 0.5f * (float) ((i * 5) % 9)); }

// one exchange added into the residual: its own rounding, in the order the pair runs the exchanges
uint16_t step_resid(uint16_t from, int e, int64_t i) {
    return kernels::bf16_from_f32(kernels::f32_from_bf16(from) + kernels::f32_from_bf16(part_a(0, e, i)) +
                                  kernels::f32_from_bf16(part_b(0, e, i)));
}

// The other form the forward uses: every exchange's sum landing ADDED INTO the running residual rows, which both cards
// must end with identical to the bit - the residual is the one stream the pair keeps replicated.
bool run_add_case(const char* what, Cfg& c, int rows) {
    const int64_t block = (int64_t) rows * kEmb;
    uint16_t *r0 = nullptr, *r1 = nullptr;
    for (int d : {c.dev0, c.dev1}) {
        const core::OnDevice on(d);
        uint16_t*& r = d == c.dev0 ? r0 : r1;
        if (cudaMalloc((void**) &r, (size_t) block * 2) != cudaSuccess) {
            check(false, std::string(what) + ": the residual rows did not fit on the card");
            return false;
        }
    }
    c.rows = rows;
    c.hp0 = host_partials(0, rows, 0);
    c.hp1 = host_partials(1, rows, 0);
    c.hp0b = c.hp0;   // the residual case runs one window; the alternating sets belong to the replay loop
    c.hp1b = c.hp1;
    if (!c.reset()) return check(false, std::string(what) + ": the partials did not reach the cards"), false;
    std::vector<uint16_t> hr((size_t) block);
    for (int64_t i = 0; i < block; ++i) hr[(size_t) i] = resid(i);
    bool up = true;
    {
        const core::OnDevice on0(c.dev0);
        up = up && cudaMemcpyAsync(r0, hr.data(), (size_t) block * 2, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    {
        const core::OnDevice on1(c.dev1);
        up = up && cudaMemcpyAsync(r1, hr.data(), (size_t) block * 2, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!up || cudaDeviceSynchronize() != cudaSuccess) return check(false, std::string(what) + ": the residual rows did not reach the cards"), false;
    for (int e = 0; e < kExch; ++e) {
        dense::tp_allreduce2_add(*c.h, 0, e, c.x0 + (int64_t) e * block, r0, rows, c.s0);
        dense::tp_allreduce2_add(*c.h, 1, e, c.x1 + (int64_t) e * block, r1, rows, c.s1);
    }
    if (!drain(c.s0, c.s1, *c.h, std::string(what) + ": window")) return false;
    dense::tp_hand_lower(*c.h);
    std::vector<uint16_t> g0((size_t) block), g1((size_t) block), p0((size_t) kExch * block), p1((size_t) kExch * block);
    bool read_ok = true;
    {
        const core::OnDevice on(c.dev0);
        read_ok = read_ok && cuda_ok(what, cudaMemcpy(g0.data(), r0, g0.size() * 2, cudaMemcpyDeviceToHost));
        read_ok = read_ok && cuda_ok(what, cudaMemcpy(p0.data(), c.x0, p0.size() * 2, cudaMemcpyDeviceToHost));
    }
    {
        const core::OnDevice on(c.dev1);
        read_ok = read_ok && cuda_ok(what, cudaMemcpy(g1.data(), r1, g1.size() * 2, cudaMemcpyDeviceToHost));
        read_ok = read_ok && cuda_ok(what, cudaMemcpy(p1.data(), c.x1, p1.size() * 2, cudaMemcpyDeviceToHost));
    }
    {
        const core::OnDevice on(c.dev0);
        cudaFree(r0);   // each card frees its own rows
    }
    {
        const core::OnDevice on(c.dev1);
        cudaFree(r1);
    }
    if (!read_ok) return false;
    int wrong = 0, same = 0, moved = 0;
    for (int64_t i = 0; i < block; ++i) {
        uint16_t w = resid(i);
        for (int e = 0; e < kExch; ++e) w = step_resid(w, e, i);
        if (g0[(size_t) i] != w || g1[(size_t) i] != w) ++wrong;
        if (g0[(size_t) i] != g1[(size_t) i]) ++same;
    }
    for (int64_t i = 0; i < (int64_t) kExch * block; ++i)
        if (p0[(size_t) i] != c.hp0[(size_t) i] || p1[(size_t) i] != c.hp1[(size_t) i]) ++moved;
    check(wrong == 0, std::string(what) + ": both cards' residual rows hold the sum of all " + std::to_string(kExch) +
                           " exchanges, each rounded once");
    check(same == 0, std::string(what) + ": the pair's residual rows stayed identical to the bit");
    check(moved == 0, std::string(what) + ": the exchanges left the partials they read alone");
    return wrong == 0 && same == 0 && moved == 0;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 2) {
        std::printf("# this test needs two cards (%d seen)\n", ndev);
        return 77;
    }
    for (int d = 0; d < 2; ++d) {   // the test's own footprint is a couple of megabytes and microseconds of work, but a card
                                    // that cannot spare even that (or cannot spare a context) is not a card to test on
        (void) cudaSetDevice(d);
        size_t free_b = 0, total = 0;
        if (cudaMemGetInfo(&free_b, &total) != cudaSuccess || free_b < (24ull << 20)) {
            std::printf("# CUDA%d has %.0f MiB free; this test wants a card with room for its context and buffers\n", d,
                        (double) free_b / 1048576.0);
            return 77;
        }
    }
    std::string err;
    Cfg c;
    c.h = dense::tp_hand_alloc(kEmb, kExch, err);
    if (!c.h) {
        std::printf("# %s\n", err.c_str());
        return 1;
    }
    bool ok = true;
    for (int d : {c.dev0, c.dev1}) {
        const core::OnDevice on(d);
        cudaStream_t& s = d == c.dev0 ? c.s0 : c.s1;
        uint16_t*& x = d == c.dev0 ? c.x0 : c.x1;
        if (cudaStreamCreate(&s) != cudaSuccess ||
            cudaMalloc((void**) &x, (size_t) kExch * kRows * kEmb * 2) != cudaSuccess)
            ok = false;
    }
    if (!ok) {   // the cards could not give even this: an environment answer, not a failing exchange
        std::printf("# the pair could not take its streams and %d MiB of buffers\n", kExch * kRows * kEmb * 2 / 1048576);
        dense::tp_hand_free(c.h);
        return 77;
    }
    check(true, "the pair's streams and buffers are there");
    run_case("a full window", c, kRows);
    run_case("a one-row window", c, kNarrow);
    run_add_case("added into the residual", c, kRows);
    dense::tp_hand_free(c.h);
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
