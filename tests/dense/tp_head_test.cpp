// tests/dense/tp_head_test.cpp - assembling a row of logits from a tensor-parallel pair's two vocabulary shards.  The
// head's product writes its matrix's out_features contiguously, so a card's shard is a COMPACT block at its own pitch
// while the logits the sampler reads are the model's whole vocabulary wide: placing a block is a pitched copy, and the
// peer's block crosses the pair's door.  Content must be exact - the token comes out of these values - and the crossing's
// cost per window row is what the head's design is judged on.  Two cards of their own; skipped without.
#include "basalt/dense/tp_geometry.hpp"
#include "tp_test_model.hpp"

#include "internal.hpp"   // TpHead, tp_peer_enable, tp_head_place (the dense stack's internals, as the exchange test's)

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

using basalt::dense::Geometry;
using basalt::dense::TpHead;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e__ = (x);                                                                                         \
        if (e__ != cudaSuccess) {                                                                                      \
            std::printf("FAIL %s: %s\n", #x, cudaGetErrorString(e__));                                                 \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

constexpr float kUntouched = -12345.0f;

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 2) {
        std::printf("SKIP: the head needs a pair of cards\n");
        return 77;
    }
    static const Geometry g = basalt::dense::tp_test::qwen38_27b();
    const int c0 = basalt::dense::tp_head_rows(g, 2, 0), c1 = basalt::dense::tp_head_rows(g, 2, 1);
    if (c0 <= 0 || c1 <= 0 || c0 + c1 != g.n_vocab) {
        std::printf("SKIP: this model's vocabulary does not share out between two cards\n");
        return 77;
    }
    const int rows = basalt::dense::kWindow;
    std::printf("#  %d window rows: card 0's %d columns and card 1's %d becoming %d-wide rows (%.2f MiB crossing the door)\n",
                rows, c0, c1, g.n_vocab, (double) rows * c1 * 4 / 1048576.0);

    std::string err;
    const bool door = basalt::dense::tp_peer_enable(0, 1, err) && basalt::dense::tp_peer_enable(1, 0, err);
    check(door, "the pair's door opens both ways" + (door ? std::string() : std::string(": ") + err));
    if (!door) {
        std::printf("the head's page-locked fallback is a different transport; this test covers the door\n");
        return 77;
    }

    // Each card's head block, compact at its own pitch; card 0's logits, whole-vocabulary wide and nothing in them yet.
    std::vector<float> own((size_t) rows * c0), peer((size_t) rows * c1);
    for (size_t r = 0; r < (size_t) rows; ++r)
        for (int c = 0; c < c0; ++c) own[(size_t) r * c0 + c] = (float) -(r * 7919 + c) * 0.5f;
    for (size_t r = 0; r < (size_t) rows; ++r)
        for (int c = 0; c < c1; ++c) peer[(size_t) r * c1 + c] = (float) (r * 1000003 + c) * 0.25f;

    // The logits are allocated with rows beyond the window on purpose: a pitched copy whose height is wrong writes the
    // next rows, and the sampler would read them a window later - the worst kind of bug to find.
    const int spare = 4;
    const int alloc_rows = rows + spare;
    float* hb0 = nullptr;   // card 0's head block
    float* hb1 = nullptr;   // card 1's, in the peer's VRAM
    float* lg = nullptr;    // the logits both cards' rows are assembled into, on card 0
    CK(cudaSetDevice(0));
    CK(cudaMalloc(&hb0, own.size() * sizeof(float)));
    CK(cudaMalloc(&lg, (size_t) alloc_rows * g.n_vocab * sizeof(float)));
    std::vector<float> fill((size_t) alloc_rows * g.n_vocab, kUntouched);
    CK(cudaMemcpy(lg, fill.data(), fill.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(hb0, own.data(), own.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaSetDevice(1));
    CK(cudaMalloc(&hb1, peer.size() * sizeof(float)));
    CK(cudaMemcpy(hb1, peer.data(), peer.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t s0 = nullptr;
    CK(cudaSetDevice(0));
    CK(cudaStreamCreate(&s0));

    TpHead own_half, peer_half;
    own_half.dst = lg;
    own_half.src = hb0;
    own_half.dst_pitch = g.n_vocab;
    own_half.src_pitch = c0;
    own_half.dst_col0 = 0;
    own_half.cols = c0;
    peer_half.dst = lg;
    peer_half.src = hb1;   // the peer's VRAM, readable because the door is open
    peer_half.dst_pitch = g.n_vocab;
    peer_half.src_pitch = c1;
    peer_half.dst_col0 = c0;
    peer_half.cols = c1;

    check(basalt::dense::tp_head_place(own_half, rows, s0, err), "the card's own block is placed" + (err.empty() ? std::string() : ": " + err));
    check(basalt::dense::tp_head_place(peer_half, rows, s0, err),
          "the peer's block is placed across the door" + (err.empty() ? std::string() : ": " + err));
    CK(cudaStreamSynchronize(s0));

    std::vector<float> back((size_t) alloc_rows * g.n_vocab, kUntouched);
    CK(cudaMemcpy(back.data(), lg, back.size() * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok_own = true, ok_peer = true, ok_spare = true;
    for (int r = 0; r < alloc_rows; ++r)
        for (int col = 0; col < g.n_vocab; ++col) {
            const float got = back[(size_t) r * g.n_vocab + col];
            if (r >= rows) {
                if (got != kUntouched) ok_spare = false;
            } else if (col < c0) {
                if (got != own[(size_t) r * c0 + col]) ok_own = false;
            } else if (got != peer[(size_t) r * c1 + (col - c0)]) {
                ok_peer = false;
            }
        }
    check(ok_own, "every one of this card's " + std::to_string(c0) + " columns of " + std::to_string(rows) +
                      " rows sits at its own column");
    check(ok_peer, "every one of the peer's " + std::to_string(c1) + " columns sits at the column that follows");
    check(ok_spare, "and the " + std::to_string(spare) + " rows past the window are untouched: the copy's height is the window");

    // The price of the crossing, per window row: this happens once per window, and its cost belongs against a token.
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    float best = 1e30f, ms = 0.f;
    for (int i = 0; i < 20; ++i) {
        CK(cudaEventRecord(a, s0));
        (void) basalt::dense::tp_head_place(peer_half, rows, s0, err);
        CK(cudaEventRecord(b, s0));
        CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
    }
    std::printf("#  crossing %d x %d columns: %.1f us = %.1f us per window row, %.2f GiB/s\n", rows, c1, best * 1000.0,
                best * 1000.0 / rows, (double) rows * c1 * 4 / 1.073741824e9 / (best / 1000.0));
    check(best * 1000.0 / rows < 200.0, "a row of the peer's vocabulary crosses well inside a millisecond");

    best = 1e30f;
    for (int i = 0; i < 20; ++i) {
        CK(cudaEventRecord(a, s0));
        (void) basalt::dense::tp_head_place(peer_half, 1, s0, err);
        CK(cudaEventRecord(b, s0));
        CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
    }
    std::printf("#  crossing one row (a session without a drafter): %.1f us\n", best * 1000.0);
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));

    // The refusals: shapes that would read or write past a block never reach the driver.
    TpHead bad = peer_half;
    bad.cols = c1 + 1;
    err.clear();
    check(!basalt::dense::tp_head_place(bad, rows, s0, err) && !err.empty(), "a block wider than the row can hold is refused");
    bad = peer_half;
    bad.dst_col0 = c0 + 1;
    err.clear();
    check(!basalt::dense::tp_head_place(bad, rows, s0, err), "and one that would land past the row's last column");
    bad = peer_half;
    bad.src_pitch = c1 - 1;
    err.clear();
    check(!basalt::dense::tp_head_place(bad, rows, s0, err), "and a block whose pitch is narrower than its own columns");
    err.clear();
    check(!basalt::dense::tp_head_place(peer_half, 0, s0, err), "and a window of no rows at all");

    CK(cudaStreamDestroy(s0));
    CK(cudaFree(hb0));
    CK(cudaFree(lg));
    CK(cudaSetDevice(1));
    CK(cudaFree(hb1));
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
