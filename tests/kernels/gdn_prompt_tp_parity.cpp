// tests/kernels/gdn_prompt_tp_parity.cpp - a tensor-parallel card's GDN slice runs as a GDN layer of its own.
//
// One 16 key / 48 value prompt GDN over T tokens, run whole; then the same layer's card slice (key heads [8, 16):
// a packed 8 key / 24 value geometry - qkv channels, z columns, gates, dt/ssm_a, conv weights and the state all
// sliced with core/gdn_head_split.hpp) run through the very same kernels with the sliced head counts.  Every head
// meets exactly one key head, so the slice's recurrence is the full layer's, head for head: the y rows of its
// value heads, its final state and conv history must be bitwise the full layer's, and its gates/bitwise-sliced.
// Exit 77 without a GPU.
#include "basalt/core/gdn_head_split.hpp"
#include "basalt/kernels/gdn_prompt.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace basalt::kernels;
using basalt::core::GdnHeadSlice;
using basalt::core::GdnRun;
using basalt::core::gdn_gather_cols;
using basalt::core::gdn_gather_rows;
using basalt::core::gdn_qkv_runs;
using basalt::core::gdn_slice_state;
using basalt::core::gdn_vhead_runs;

namespace {

int fails = 0;
std::mt19937 rng(20261118);
std::normal_distribution<float> nd(0.f, 1.f);

std::vector<float> randf(size_t n, float sd) {
    std::vector<float> v(n);
    for (auto& x : v) x = sd * nd(rng);
    return v;
}
std::vector<uint16_t> randbf(size_t n, float sd) {
    std::vector<uint16_t> v(n);
    for (auto& x : v) {
        const float f = sd * nd(rng);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        x = (uint16_t) (u >> 16);
    }
    return v;
}

template <class T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess) {
        std::printf("  cudaMalloc failed\n");
        std::exit(1);
    }
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

bool bytes_same(const char* what, const void* a, const void* b, size_t bytes) {
    std::vector<unsigned char> x(bytes), y(bytes);
    cudaStreamSynchronize(0);
    cudaMemcpy(x.data(), a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), b, bytes, cudaMemcpyDeviceToHost);
    size_t d = 0, first = 0;
    for (size_t i = 0; i < bytes; ++i)
        if (x[i] != y[i]) {
            if (!d) first = i;
            ++d;
        }
    if (d) std::printf("  FAIL %s: %zu of %zu bytes differ (first at %zu)\n", what, d, bytes, first);
    else std::printf("  ok %s\n", what);
    fails += (d != 0);
    return d == 0;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::printf("no GPU - skipping\n");
        return 77;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        std::printf("no GPU - skipping\n");
        return 77;
    }

    constexpr int S = 128, HK = 16, HV = 48;
    constexpr int k0 = 8, NK = 8;                 // the card: key heads [8, 16)
    constexpr int NV = NK * (HV / HK);            // 24 value heads, packed j*NK + k'
    constexpr int64_t T = 100;                    // not a multiple of the 16-token chunk
    const int C = (2 * HK + HV) * S, CS = (2 * NK + NV) * S;

    if (!gdn_prompt_supported(S, HK, HV, 4) || !gdn_prompt_supported(S, NK, NV, 4)) {
        std::printf("FAIL gdn_prompt_supported rejects the full or the 8/24 slice\n");
        return 1;
    }

    GdnHeadSlice g{HK, HV, S, k0, NK};
    const std::vector<GdnRun> qr = gdn_qkv_runs(g), vr1 = gdn_vhead_runs(g, 1), vs = gdn_vhead_runs(g, S);

    // ---- the full layer's inputs
    std::vector<uint16_t> qkv = randbf((size_t) T * C, 1.f), z = randbf((size_t) T * HV * S, 1.f);
    std::vector<float> ab = randf((size_t) T * 2 * HV, 1.f), dt = randf(HV, 0.5f), ssm_a = randf(HV, 0.5f);
    for (auto& x : ssm_a) x = -std::abs(x) - 0.01f;   // decay like the model's
    std::vector<float> gamma = randf(S, 1.f), conv_w = randf((size_t) C * 4, 0.5f);
    std::vector<float> hist_full = randf((size_t) C * 3, 0.5f);
    std::vector<float> state_full((size_t) S * HV * S, 0.f);

    // ---- the slice's inputs, packed
    std::vector<uint16_t> qkv_s((size_t) T * CS), z_s((size_t) T * NV * S);
    gdn_gather_cols(qkv.data(), T, C, S, (size_t) S * 2, qr, qkv_s.data());
    gdn_gather_cols(z.data(), T, HV * S, S, (size_t) S * 2, vs, z_s.data());
    std::vector<float> ab_s((size_t) T * 2 * NV);
    for (int64_t t = 0; t < T; ++t)
        for (size_t j = 0; j < vr1.size(); ++j)
            for (int64_t i = 0; i < NK; ++i) {
                ab_s[(size_t) t * 2 * NV + (size_t) j * NK + i] = ab[(size_t) t * 2 * HV + vr1[j].start + i];
                ab_s[(size_t) t * 2 * NV + NV + (size_t) j * NK + i] = ab[(size_t) t * 2 * HV + HV + vr1[j].start + i];
            }
    std::vector<float> dt_s(NV), ssm_a_s(NV);
    for (size_t j = 0; j < vr1.size(); ++j)
        for (int64_t i = 0; i < NK; ++i) {
            dt_s[(size_t) j * NK + i] = dt[vr1[j].start + i];
            ssm_a_s[(size_t) j * NK + i] = ssm_a[vr1[j].start + i];
        }
    std::vector<float> conv_w_s((size_t) CS * 4);
    gdn_gather_rows(conv_w.data(), sizeof(float) * 4, qr, conv_w_s.data());
    std::vector<float> combined_full(((size_t) S * HV * S) + (size_t) C * 3);
    std::memcpy(combined_full.data(), state_full.data(), sizeof(float) * state_full.size());
    std::memcpy(combined_full.data() + state_full.size(), hist_full.data(), sizeof(float) * hist_full.size());
    std::vector<float> combined_s(gdn_slice_state_floats(g, 3));
    gdn_slice_state(g, 3, combined_full.data(), combined_s.data());
    std::vector<float> state_s(combined_s.begin(), combined_s.begin() + (size_t) S * NV * S);
    std::vector<float> hist_s(combined_s.begin() + (size_t) S * NV * S, combined_s.end());

    // ---- device
    uint16_t *dqkv = upload(qkv), *dqkvs = upload(qkv_s), *dz = upload(z), *dzs = upload(z_s);
    float *dab = upload(ab), *dabs = upload(ab_s), *ddt = upload(dt), *ddts = upload(dt_s);
    float *dssm = upload(ssm_a), *dssms = upload(ssm_a_s), *dgamma = upload(gamma);
    float *dcw = upload(conv_w), *dcws = upload(conv_w_s);
    float *dhf = upload(hist_full), *dhs = upload(hist_s);
    float *dsf = upload(state_full), *dss = upload(state_s);
    float *gbuf = nullptr, *bbuf = nullptr, *gsbuf = nullptr, *bsbuf = nullptr;
    cudaMalloc(&gbuf, (size_t) T * HV * 4);
    cudaMalloc(&bbuf, (size_t) T * HV * 4);
    cudaMalloc(&gsbuf, (size_t) T * NV * 4);
    cudaMalloc(&bsbuf, (size_t) T * NV * 4);
    uint16_t *hf_f = nullptr, *hf_s = nullptr;
    cudaMalloc(&hf_f, (size_t) T * C * 2);
    cudaMalloc(&hf_s, (size_t) T * CS * 2);
    float *y_f = nullptr, *y_s = nullptr;
    cudaMalloc(&y_f, (size_t) T * HV * S * 4);
    cudaMalloc(&y_s, (size_t) T * NV * S * 4);
    std::vector<char> work_f(gdn_work_bytes(T, HK, HV)), work_s(gdn_work_bytes(T, NK, NV));
    if (cudaGetLastError() != cudaSuccess) {
        std::printf("FAIL device setup: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    // ---- full layer, start to finish
    gdn_gates(dab, ddt, dssm, gbuf, bbuf, T, 0, HK, HV);
    gdn_conv(dhf, dqkv, dcw, hf_f, T, 0, HK, HV);
    gdn_recurrence_bf16(dsf, hf_f, gbuf, bbuf, dz, dgamma, 1e-6f, work_f.data(), y_f, T, 0, GdnGate::Sigmoid,
                        nullptr, nullptr, nullptr, nullptr, HK, HV);
    if (cudaStreamSynchronize(0) != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        std::printf("FAIL full-layer run: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    // ---- the card's slice, same kernels, its own head counts
    gdn_gates(dabs, ddts, dssms, gsbuf, bsbuf, T, 0, NK, NV);
    gdn_conv(dhs, dqkvs, dcws, hf_s, T, 0, NK, NV);
    gdn_recurrence_bf16(dss, hf_s, gsbuf, bsbuf, dzs, dgamma, 1e-6f, work_s.data(), y_s, T, 0, GdnGate::Sigmoid,
                        nullptr, nullptr, nullptr, nullptr, NK, NV);
    if (cudaStreamSynchronize(0) != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        std::printf("FAIL slice run: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    // ---- the slice's gates are the full layer's gates, head for head
    std::vector<float> gate_host((size_t) T * HV), gate_card((size_t) T * NV);
    cudaMemcpy(gate_host.data(), gbuf, sizeof(float) * gate_host.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(gate_card.data(), gsbuf, sizeof(float) * gate_card.size(), cudaMemcpyDeviceToHost);
    {
        size_t bad = 0;
        for (int64_t t = 0; t < T; ++t)
            for (size_t j = 0; j < vr1.size(); ++j)
                for (int64_t i = 0; i < NK; ++i) {
                    const float a = gate_host[(size_t) t * HV + vr1[j].start + i];
                    const float b = gate_card[(size_t) t * NV + (size_t) j * NK + i];
                    bad += std::memcmp(&a, &b, 4) != 0;
                }
        if (bad) std::printf("  FAIL gates: %zu differ\n", bad);
        else std::printf("  ok gates\n");
        fails += (bad != 0);
    }

    // ---- y, bitwise, per value head (with the exact packed-local -> global head map)
    {
        auto global_head = [&](int local) { return (local / NK) * HK + k0 + (local % NK); };
        std::vector<float> yf((size_t) T * HV * S), ys((size_t) T * NV * S);
        cudaMemcpy(yf.data(), y_f, sizeof(float) * yf.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(ys.data(), y_s, sizeof(float) * ys.size(), cudaMemcpyDeviceToHost);
        size_t bad = 0;
        for (int64_t t = 0; t < T; ++t)
            for (int h = 0; h < NV; ++h)
                if (std::memcmp(&ys[(size_t) t * NV * S + (size_t) h * S], &yf[(size_t) t * HV * S + (size_t) global_head(h) * S],
                                sizeof(float) * S))
                    ++bad;
        if (bad) std::printf("  FAIL y: %zu of %d head-rows differ\n", bad, NV);
        else std::printf("  ok y (%d heads x %lld tokens, bitwise)\n", NV, (long long) T);
        fails += (bad != 0);
    }

    // ---- the final state and conv history: the sliced end state equals the slice of the full end state
    {
        std::vector<float> end_full(combined_full.size()), end_s(combined_s.size());
        cudaMemcpy(end_full.data(), dsf, sizeof(float) * state_full.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(end_full.data() + state_full.size(), dhf, sizeof(float) * hist_full.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(end_s.data(), dss, sizeof(float) * state_s.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(end_s.data() + state_s.size(), dhs, sizeof(float) * hist_s.size(), cudaMemcpyDeviceToHost);
        std::vector<float> want(combined_s.size());
        gdn_slice_state(g, 3, end_full.data(), want.data());
        size_t bad = 0;
        for (size_t i = 0; i < want.size(); ++i) bad += std::memcmp(&want[i], &end_s[i], 4) != 0;
        if (bad) std::printf("  FAIL end state/history: %zu of %zu floats differ\n", bad, want.size());
        else std::printf("  ok end state + conv history (%zu floats, bitwise)\n", want.size());
        fails += (bad != 0);
    }

    std::printf(fails ? "FAILED (%d)\n" : "all ok\n", fails);
    return fails ? 1 : 0;
}
