// tests/core/tp_mailbox_test.cpp - the two-card mailbox exchange gives the same bits on both cards for every
// payload shape (BF16 sum, fused residual add, FP32 sum), survives CUDA-graph capture and replay, refuses a
// round whose cards disagree on its geometry, finishes a lost-peer exchange by reporting the hang word, and
// carries each payload size through a loose time bound.  Every round enqueues rank 1 first - the protocol's
// order (tp_mailbox.cu's n_agree: rank 1 writes the slot's geometry, rank 0's compares), and the graphs are
// captured and launched rank 1 first as the pair's drivers do.  Needs two GPUs; exits 77 without them.
#include "basalt/core/tp_mailbox.hpp"
#include "basalt/core/vmm.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace basalt::core;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}

// Which payload path a mailbox opened with: device VMM or the pinned block.
const char* path_name(bool device) { return device ? "device" : "pinned"; }

// The host model of the combines, in the same rounding order as the kernels.
uint16_t rn_bf16(float f) {
    __nv_bfloat16 h = __float2bfloat16_rn(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
float to_f(uint16_t u) {
    __nv_bfloat16 h;
    std::memcpy(&h, &u, 2);
    return __bfloat162float(h);
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 2) {
        std::printf("SKIP: needs two GPUs\n");
        return 77;
    }

    const size_t kBytes = 1 << 20;  // the largest slot payload the mailbox opens with
    const size_t kSlots = 16;
    TpMailbox mb;
    std::string err;
    if (!mb.open(0, 1, kSlots, kBytes, err)) {
        std::printf("FAIL: open: %s\n", err.c_str());
        return 1;
    }
    std::printf("note: payload path = %s (slot %zu B)\n", path_name(mb.device_payload()), mb.slot_bytes());
    std::printf("note: vmm supported=%d gran=%llu\n", (int) basalt::core::vmm::supported(),
                (unsigned long long) basalt::core::vmm::granularity());
    mb.trace("device-path open");
    cudaStream_t s0, s1;
    cudaSetDevice(0);
    cudaStreamCreate(&s0);
    cudaSetDevice(1);
    cudaStreamCreate(&s1);

    std::mt19937 rng(1234);
    const std::vector<size_t> vec_counts = {8, 64, 640, 10240};  // 16 B .. 160 KiB of BF16 payload
    size_t slot = 0;
    for (size_t vecs : vec_counts) {
        const size_t nbf = vecs * 8;
        std::vector<uint16_t> h0(nbf), h1(nbf);
        for (size_t i = 0; i < nbf; ++i) {
            h0[i] = rn_bf16((rng() / float(rng.max())) * 4.0f - 2.0f);
            h1[i] = rn_bf16((rng() / float(rng.max())) * 4.0f - 2.0f);
        }
        uint16_t *p0, *p1;
        cudaSetDevice(0);
        cudaMalloc(&p0, nbf * 2);
        cudaMemcpy(p0, h0.data(), nbf * 2, cudaMemcpyHostToDevice);
        cudaSetDevice(1);
        cudaMalloc(&p1, nbf * 2);
        cudaMemcpy(p1, h1.data(), nbf * 2, cudaMemcpyHostToDevice);
        // a few rounds in a row: every round replaces both cards' values with their sum's rounding
        std::vector<uint16_t> a = h0, b = h1;
        bool ok = true;
        for (int round = 0; round < 4 && ok; ++round) {
            slot = (slot + 1) % kSlots;
            cudaSetDevice(1);
            if (!mb.exchange_bf16(1, slot, p1, nbf * 2, s1, err)) { ok = false; break; }
            cudaSetDevice(0);
            if (!mb.exchange_bf16(0, slot, p0, nbf * 2, s0, err)) { ok = false; break; }
            cudaStreamSynchronize(s1);
            cudaStreamSynchronize(s0);
            for (size_t i = 0; i < nbf; ++i) {
                const uint16_t s = rn_bf16(to_f(a[i]) + to_f(b[i]));
                a[i] = b[i] = s;
            }
        }
        std::vector<uint16_t> r0(nbf), r1(nbf);
        cudaSetDevice(0);
        cudaMemcpy(r0.data(), p0, nbf * 2, cudaMemcpyDeviceToHost);
        cudaSetDevice(1);
        cudaMemcpy(r1.data(), p1, nbf * 2, cudaMemcpyDeviceToHost);
        ok = ok && r0 == a && r1 == b && r0 == r1;
        check(ok, "bf16 sum exchange of " + std::to_string(nbf * 2) + " bytes over 4 rounds");
        cudaSetDevice(0);
        cudaFree(p0);
        cudaSetDevice(1);
        cudaFree(p1);
        mb.close();   // the shape's slots are 2 MiB per slot: give them back before the next shape opens its own
        if (!mb.open(0, 1, 16, kBytes, err)) { std::printf("FAIL: reopen: %s\n", err.c_str()); return 1; }
        }

    // the fused residual add: base = bf16(fp32(base) + fp32(bf16(mine0)) + fp32(bf16(mine1)))
    {
        const size_t nval = 640 * 8;
        std::vector<float> m0(nval), m1(nval);
        std::vector<uint16_t> base(nval);
        for (size_t i = 0; i < nval; ++i) {
            m0[i] = (rng() / float(rng.max())) * 4.0f - 2.0f;
            m1[i] = (rng() / float(rng.max())) * 4.0f - 2.0f;
            base[i] = rn_bf16((rng() / float(rng.max())) * 8.0f - 4.0f);
        }
        std::vector<uint16_t> want(nval);
        for (size_t i = 0; i < nval; ++i)
            want[i] = rn_bf16(to_f(base[i]) + to_f(rn_bf16(m0[i])) + to_f(rn_bf16(m1[i])));
        float *d0, *d1;
        uint16_t *b0, *b1;
        cudaSetDevice(0);
        cudaMalloc(&d0, nval * 4);
        cudaMalloc(&b0, nval * 2);
        cudaMemcpy(d0, m0.data(), nval * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(b0, base.data(), nval * 2, cudaMemcpyHostToDevice);
        cudaSetDevice(1);
        cudaMalloc(&d1, nval * 4);
        cudaMalloc(&b1, nval * 2);
        cudaMemcpy(d1, m1.data(), nval * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(b1, base.data(), nval * 2, cudaMemcpyHostToDevice);
        bool ok = true;
        for (int round = 0; round < 3; ++round) {
            slot = (slot + 1) % kSlots;
            cudaSetDevice(1);
            ok = ok && mb.exchange_add_bf16(1, slot, d1, b1, nval, s1, err);
            cudaSetDevice(0);
            ok = ok && mb.exchange_add_bf16(0, slot, d0, b0, nval, s0, err);
            cudaStreamSynchronize(s1);
            cudaStreamSynchronize(s0);
            for (size_t i = 0; i < nval; ++i)
                want[i] = base[i] = rn_bf16(to_f(base[i]) + to_f(rn_bf16(m0[i])) + to_f(rn_bf16(m1[i])));
        }
        std::vector<uint16_t> r(nval);
        cudaSetDevice(0);
        cudaMemcpy(r.data(), b0, nval * 2, cudaMemcpyDeviceToHost);
        ok = ok && r == want;
        cudaSetDevice(1);
        cudaMemcpy(r.data(), b1, nval * 2, cudaMemcpyDeviceToHost);
        ok = ok && r == want;
        check(ok, "fused residual-add exchange over 3 rounds, equal bits on both cards");
        cudaSetDevice(0);
        cudaFree(d0);
        cudaFree(b0);
        cudaSetDevice(1);
        cudaFree(d1);
        cudaFree(b1);
    }

    // the FP32 sum (exact): with and without the BF16 side output
    {
        const size_t nval = 1024 * 4;
        std::vector<float> f0(nval), f1(nval);
        for (size_t i = 0; i < nval; ++i) {
            f0[i] = (rng() / float(rng.max())) * 4.0f - 2.0f;
            f1[i] = (rng() / float(rng.max())) * 4.0f - 2.0f;
        }
        std::vector<float> want(nval);
        std::vector<uint16_t> want16(nval), got16(nval);
        for (size_t i = 0; i < nval; ++i) {
            want[i] = f0[i] + f1[i];
            want16[i] = rn_bf16(want[i]);
        }
        float *d0, *d1;
        uint16_t *o0, *o1;
        cudaSetDevice(0);
        cudaMalloc(&d0, nval * 4);
        cudaMalloc(&o0, nval * 2);
        cudaMemcpy(d0, f0.data(), nval * 4, cudaMemcpyHostToDevice);
        cudaSetDevice(1);
        cudaMalloc(&d1, nval * 4);
        cudaMalloc(&o1, nval * 2);
        cudaMemcpy(d1, f1.data(), nval * 4, cudaMemcpyHostToDevice);
        slot = (slot + 1) % kSlots;
        bool ok = true;
        cudaSetDevice(1);
        ok = ok && mb.exchange_f32(1, slot, d1, o1, nval, s1, err);
        cudaSetDevice(0);
        ok = ok && mb.exchange_f32(0, slot, d0, o0, nval, s0, err);
        cudaStreamSynchronize(s1);
        cudaStreamSynchronize(s0);
        std::vector<float> r(nval);
        cudaSetDevice(0);
        cudaMemcpy(r.data(), d0, nval * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(got16.data(), o0, nval * 2, cudaMemcpyDeviceToHost);
        ok = ok && r == want && got16 == want16;
        cudaSetDevice(1);
        cudaMemcpy(r.data(), d1, nval * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(got16.data(), o1, nval * 2, cudaMemcpyDeviceToHost);
        ok = ok && r == want && got16 == want16;
        check(ok, "fp32 sum exchange with BF16 output, exact on both cards");
        cudaSetDevice(0);
        cudaFree(d0);
        cudaFree(o0);
        cudaSetDevice(1);
        cudaFree(d1);
        cudaFree(o1);
    }

    // CUDA graph capture: four chained exchanges of alternating slots, captured rank 1 first (the drivers'
    // order: the deep card's graph exists before the other's enqueue), replayed twice, launched rank 1 first.
    {
        const size_t nbf = 640 * 8;
        std::vector<uint16_t> h0(nbf), h1(nbf);
        for (size_t i = 0; i < nbf; ++i) {
            h0[i] = rn_bf16((rng() / float(rng.max())) * 4.0f - 2.0f);
            h1[i] = rn_bf16((rng() / float(rng.max())) * 4.0f - 2.0f);
        }
        std::vector<uint16_t> a = h0, b = h1;
        uint16_t *p0, *p1;
        cudaSetDevice(0);
        cudaMalloc(&p0, nbf * 2);
        cudaMemcpy(p0, h0.data(), nbf * 2, cudaMemcpyHostToDevice);
        cudaSetDevice(1);
        cudaMalloc(&p1, nbf * 2);
        cudaMemcpy(p1, h1.data(), nbf * 2, cudaMemcpyHostToDevice);
        cudaGraphExec_t g0 = nullptr, g1 = nullptr;
        bool ok = true;
        cudaGraph_t gr;
        cudaSetDevice(1);
        ok = ok && cudaStreamBeginCapture(s1, cudaStreamCaptureModeRelaxed) == cudaSuccess;
        for (int k = 0; k < 4; ++k)
            ok = ok && mb.exchange_bf16(1, size_t(k), p1, nbf * 2, s1, err);
        ok = ok && cudaStreamEndCapture(s1, &gr) == cudaSuccess && cudaGraphInstantiate(&g1, gr, 0) == cudaSuccess;
        cudaGraphDestroy(gr);
        cudaSetDevice(0);
        ok = ok && cudaStreamBeginCapture(s0, cudaStreamCaptureModeRelaxed) == cudaSuccess;
        for (int k = 0; k < 4; ++k)
            ok = ok && mb.exchange_bf16(0, size_t(k), p0, nbf * 2, s0, err);
        ok = ok && cudaStreamEndCapture(s0, &gr) == cudaSuccess && cudaGraphInstantiate(&g0, gr, 0) == cudaSuccess;
        cudaGraphDestroy(gr);
        for (int replay = 0; replay < 2 && ok; ++replay) {
            cudaSetDevice(1);
            ok = ok && cudaGraphLaunch(g1, s1) == cudaSuccess;
            cudaSetDevice(0);
            ok = ok && cudaGraphLaunch(g0, s0) == cudaSuccess;
            cudaSetDevice(1);
            cudaStreamSynchronize(s1);
            cudaSetDevice(0);
            cudaStreamSynchronize(s0);
            for (int k = 0; k < 4; ++k)
                for (size_t i = 0; i < nbf; ++i) a[i] = b[i] = rn_bf16(to_f(a[i]) + to_f(b[i]));
        }
        std::vector<uint16_t> r0(nbf), r1(nbf);
        cudaSetDevice(0);
        cudaMemcpy(r0.data(), p0, nbf * 2, cudaMemcpyDeviceToHost);
        cudaSetDevice(1);
        cudaMemcpy(r1.data(), p1, nbf * 2, cudaMemcpyDeviceToHost);
        ok = ok && r0 == a && r1 == b && r0 == r1;
        check(ok, "four captured exchanges replayed twice keep equal bits on both cards");
        cudaGraphExecDestroy(g0);
        cudaGraphExecDestroy(g1);
        cudaSetDevice(0);
        cudaFree(p0);
        cudaSetDevice(1);
        cudaFree(p1);
    }

    // latency and bandwidth: median over consecutive exchanges at the biggest useful payloads.  Both paths, so a
    // device-payload mailbox can be priced against the pinned one it replaces.
    for (int path = 0; path < 2; ++path) {
        const bool device = path == 0;
        TpMailbox* mbp = device ? &mb : nullptr;
        TpMailbox pin;
        if (!device) {
            if (!pin.open(0, 1, 16, kBytes, err, false)) { check(false, "the pinned latency mailbox opens: " + err); break; }
            mbp = &pin;
        }
        TpMailbox& m = *mbp;
        const std::vector<std::pair<size_t, double>> bounds = {{40960, 200.0}, {163840, 400.0}, {1048576, 3000.0}};
        for (auto [bytes, us_bound] : bounds) {
            const size_t nbf = bytes / 2;
            uint16_t *p0, *p1;
            cudaSetDevice(0);
            cudaMalloc(&p0, bytes);
            cudaMemset(p0, 0x3f, bytes);
            cudaSetDevice(1);
            cudaMalloc(&p1, bytes);
            cudaMemset(p1, 0x3f, bytes);
            std::vector<double> times;
            for (int k = 0; k < 24; ++k) {
                slot = (slot + 1) % kSlots;
                cudaSetDevice(1);
                m.exchange_bf16(1, slot, p1, bytes, s1, err);
                auto t0 = std::chrono::steady_clock::now();
                cudaSetDevice(0);
                m.exchange_bf16(0, slot, p0, bytes, s0, err);
                cudaStreamSynchronize(s0);
                cudaStreamSynchronize(s1);
                times.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            const double med = median(times);
            std::printf("note: %s exchange of %zu B: median %.1f us (%.2f GB/s per card)\n", path_name(device), bytes,
                        med, bytes / med / 1000.0);
            check(med < us_bound, std::string(path_name(device)) + ": exchange of " + std::to_string(bytes) + " B under the bound");
            cudaSetDevice(0);
            cudaFree(p0);
            cudaSetDevice(1);
            cudaFree(p1);
        }
        if (!device) pin.close();
    }

    // geometry disagreement: rank 1 says one payload size, rank 0 another - rank 0's call must refuse before
    // it enqueues (n_agree), and the unpaired rank-1 kernel reports the hang once its spin budget runs out.
    // Its own mailbox: the hang word is sticky, and the sections above must keep turning.
    {
        TpMailbox bad;
        if (!bad.open(0, 1, 2, kBytes, err)) {
            check(false, "the disagreement mailbox opens: " + err);
        } else {
            const size_t na = 64 * 8, nb = 640 * 8;
            uint16_t *pa, *pb;
            cudaSetDevice(1);
            cudaMalloc(&pa, na * 2);
            cudaMemset(pa, 0, na * 2);
            cudaSetDevice(0);
            cudaMalloc(&pb, nb * 2);
            cudaMemset(pb, 0, nb * 2);
            cudaSetDevice(1);
            const bool ok1 = bad.exchange_bf16(1, 0, pa, na * 2, s1, err);
            cudaSetDevice(0);
            const bool ok0 = bad.exchange_bf16(0, 0, pb, nb * 2, s0, err);
            check(ok1 && !ok0 && err.find("disagree") != std::string::npos,
                  "a round whose cards disagree on the payload is refused: " + err);
            cudaStreamSynchronize(s1);   // the unpaired rank-1 kernel spins out its budget (~2 s)
            cudaStreamSynchronize(s0);
            check(bad.hang_reported(), "the unpaired round reports the hang");
            cudaSetDevice(1);
            cudaFree(pa);
            cudaSetDevice(0);
            cudaFree(pb);
            bad.close();
        }
    }

    // lost peer: one card's exchange alone must give up and set the hang word (a fresh mailbox: the word is
    // sticky, and a slot the pair already used would carry its geometry).
    {
        TpMailbox lost;
        if (!lost.open(0, 1, 1, kBytes, err)) {
            check(false, "the lost-peer mailbox opens: " + err);
        } else {
            const size_t nbf = 64 * 8;
            uint16_t* p0;
            cudaSetDevice(0);
            cudaMalloc(&p0, nbf * 2);
            cudaMemset(p0, 0, nbf * 2);
            const auto t0 = std::chrono::steady_clock::now();
            lost.exchange_bf16(0, 0, p0, nbf * 2, s0, err);
            cudaStreamSynchronize(s0);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            check(lost.hang_reported(), "a peerless exchange reports the hang");
            check(secs < 30.0, "the peerless exchange gave up promptly");
            std::printf("note: peerless exchange gave up after %.2f s\n", secs);
            cudaFree(p0);
            lost.close();
        }
    }

    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);
    mb.close();

    // the pinned path, on the same shapes: both paths must give the same bits, and the fallback must still open
    {
        TpMailbox pin;
        if (!pin.open(0, 1, 16, kBytes, err, false)) {
            check(false, "the pinned mailbox opens: " + err);
        } else {
            check(!pin.device_payload(), "want_device=false gives the pinned path");
            cudaStream_t q0, q1;   // fresh streams: the device path's open/close moved the current device around
            cudaSetDevice(0);
            cudaStreamCreate(&q0);
            cudaSetDevice(1);
            cudaStreamCreate(&q1);
            const size_t nbf = 640 * 8;
            std::vector<uint16_t> h0(nbf), h1(nbf);
            for (size_t i = 0; i < nbf; ++i) {
                h0[i] = rn_bf16((rng() / float(rng.max())) * 4.0f - 2.0f);
                h1[i] = rn_bf16((rng() / float(rng.max())) * 4.0f - 2.0f);
            }
            std::vector<uint16_t> a = h0, b = h1;
            uint16_t *p0, *p1;
            cudaSetDevice(0);
            cudaMalloc(&p0, nbf * 2);
            cudaMemcpy(p0, h0.data(), nbf * 2, cudaMemcpyHostToDevice);
            cudaSetDevice(1);
            cudaMalloc(&p1, nbf * 2);
            cudaMemcpy(p1, h1.data(), nbf * 2, cudaMemcpyHostToDevice);
            bool ok = true;
            for (int round = 0; round < 4 && ok; ++round) {
                slot = (slot + 1) % kSlots;
                cudaSetDevice(1);
                ok = ok && pin.exchange_bf16(1, slot, p1, nbf * 2, q1, err);
                cudaSetDevice(0);
                ok = ok && pin.exchange_bf16(0, slot, p0, nbf * 2, q0, err);
                cudaStreamSynchronize(q1);
                cudaStreamSynchronize(q0);
                for (size_t i = 0; i < nbf; ++i) {
                    const uint16_t s = rn_bf16(to_f(a[i]) + to_f(b[i]));
                    a[i] = b[i] = s;
                }
            }
            std::vector<uint16_t> r0(nbf), r1(nbf);
            cudaSetDevice(0);
            cudaMemcpy(r0.data(), p0, nbf * 2, cudaMemcpyDeviceToHost);
            cudaSetDevice(1);
            cudaMemcpy(r1.data(), p1, nbf * 2, cudaMemcpyDeviceToHost);
            ok = ok && r0 == a && r1 == b && r0 == r1;
            check(ok, "pinned path: bf16 sum exchange of 10 KiB over 4 rounds");
            cudaSetDevice(0);
            cudaFree(p0);
            cudaSetDevice(1);
            cudaFree(p1);
            cudaStreamDestroy(q0);
            cudaStreamDestroy(q1);
            pin.close();
        }
    }

    std::printf("%s: %d failures\n", failures ? "FAIL" : "ok", failures);
    return failures ? 1 : 0;
}
