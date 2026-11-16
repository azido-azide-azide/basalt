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
// The flags are lowered between windows only after both cards' streams are idle, which is the discipline `TpHand` states.
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

// the values the two cards start each exchange with, and the sum the exchange owes them
uint16_t part_a(int e, int64_t i) { return kernels::bf16_from_f32(0.5f + (float) (i % 13) + 0.25f * (float) e); }
uint16_t part_b(int e, int64_t i) { return kernels::bf16_from_f32(-0.125f + (float) ((i * 7) % 11) - 0.5f * (float) e); }
uint16_t want(int e, int64_t i) {
    return kernels::bf16_from_f32(kernels::f32_from_bf16(part_a(e, i)) + kernels::f32_from_bf16(part_b(e, i)));
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

std::vector<uint16_t> host_partials(int rank, int rows) {
    std::vector<uint16_t> h((size_t) kExch * rows * kEmb);
    for (int e = 0; e < kExch; ++e)
        for (int64_t i = 0; i < (int64_t) rows * kEmb; ++i)
            h[(size_t) e * rows * kEmb + i] = rank == 0 ? part_a(e, i) : part_b(e, i);
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
bool check_sums(int dev0, int dev1, const uint16_t* x0, const uint16_t* x1, int rows, const std::string& at) {
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
            const uint16_t w = want(e, i);
            if (got0[(size_t) e * block + i] != w) { ++wrong0; if (first0 < 0) first0 = ((int64_t) e << 40) | i; }
            if (got1[(size_t) e * block + i] != w) { ++wrong1; if (first1 < 0) first1 = ((int64_t) e << 40) | i; }
        }
    if (wrong0 || wrong1) {   // the first difference on each card, spelled out
        const int e0 = (int) (first0 >> 40);
        const int64_t i0 = first0 & ((1ll << 40) - 1);
        const int e1 = (int) (first1 >> 40);
        const int64_t i1 = first1 & ((1ll << 40) - 1);
        std::printf("#   card 0: %d wrong (first exchange %d value %lld: %.6f, wants %.6f)\n", wrong0, e0, (long long) i0,
                    wrong0 ? kernels::f32_from_bf16(got0[(size_t) e0 * block + i0]) : 0.0f, kernels::f32_from_bf16(want(e0, i0)));
        std::printf("#   card 1: %d wrong (first exchange %d value %lld: %.6f, wants %.6f)\n", wrong1, e1, (long long) i1,
                    wrong1 ? kernels::f32_from_bf16(got1[(size_t) e1 * block + i1]) : 0.0f, kernels::f32_from_bf16(want(e1, i1)));
    }
    check(wrong0 == 0, at + ": card 0's every exchange holds the two cards' sum");
    check(wrong1 == 0, at + ": card 1's every exchange holds the two cards' sum");
    return wrong0 == 0 && wrong1 == 0;
}

struct Cfg {
    int dev0 = 0, dev1 = 1;
    TpHand* h = nullptr;
    uint16_t *x0 = nullptr, *x1 = nullptr;
    cudaStream_t s0 = nullptr, s1 = nullptr;
    std::vector<uint16_t> hp0, hp1;
    int rows = kRows;

    bool reset() {   // the exchanges' partials again, both cards idle
        return fill(dev0, x0, hp0, rows) && fill(dev1, x1, hp1, rows);
    }
};

bool run_case(const char* what, Cfg& c, int rows) {
    c.rows = rows;
    c.hp0 = host_partials(0, rows);
    c.hp1 = host_partials(1, rows);
    if (!c.reset()) return check(false, std::string(what) + ": the partials did not reach the cards"), false;
    // as launched
    run_window(*c.h, 0, c.s0, c.x0, rows);
    run_window(*c.h, 1, c.s1, c.x1, rows);
    if (!drain(c.s0, c.s1, *c.h, std::string(what) + ": window")) return false;
    check_sums(c.dev0, c.dev1, c.x0, c.x1, rows, std::string(what) + ": launched");
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
    for (int r = 0; r < kReplays; ++r) {
        if (!c.reset()) return check(false, std::string(what) + ": a replay's partials did not reach the cards"), false;
        if (cudaGraphLaunch(g1, c.s1) != cudaSuccess || cudaGraphLaunch(g0, c.s0) != cudaSuccess) {
            check(false, std::string(what) + ": a replay did not launch");
            return false;
        }
        if (!drain(c.s0, c.s1, *c.h, std::string(what) + ": replay " + std::to_string(r))) return false;
        dense::tp_hand_lower(*c.h);
        if (r == 0 || r == kReplays - 1) {
            const int64_t block = (int64_t) rows * kEmb;
            std::vector<uint16_t> got((size_t) kExch * block);
            (void) cudaSetDevice(c.dev0);
            if (cudaMemcpy(got.data(), c.x0, got.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            for (int e = 0; r == 0 && e < kExch; ++e)
                for (int64_t i = 0; i < block; ++i)
                    if (got[(size_t) e * block + i] != want(e, i)) { ++bad; break; }
            if (bad) break;
        }
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    check(bad == 0, std::string(what) + ": every replay of the captured window gave the sums again");
    std::printf("# %s: %d replays of %d exchanges in %d rows: %.1f us per window, %.2f us per exchange\n", what, kReplays, kExch,
                rows, us / kReplays, us / (kReplays * (double) kExch));
    cudaGraphExecDestroy(g0);
    cudaGraphExecDestroy(g1);
    return bad == 0;
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
    dense::tp_hand_free(c.h);
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
