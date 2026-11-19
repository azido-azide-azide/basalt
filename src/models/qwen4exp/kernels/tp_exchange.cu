// src/models/qwen4exp/kernels/tp_exchange.cu - see the header. Plain launches: the pair's exchange ordering is
// the stream's, and TpMailbox's own kernels keep their per-slot epochs.
#include "basalt/models/qwen4exp/kernels/tp_exchange.hpp"

#include <cuda_bf16.h>

namespace basalt::kernels {
namespace {

constexpr int kTpThreads = 256;

__global__ void tp_place_f32_kernel(uint16_t* __restrict__ full, const float* __restrict__ src, int64_t half,
                                    int64_t total, int64_t off, int64_t n_total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    const int64_t t = i / total, c = i - t * total;
    float v = 0.f;
    if (c >= off && c < off + half) v = src[t * half + (c - off)];
    full[i] = (uint16_t) __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

__global__ void tp_place_bf_kernel(uint16_t* __restrict__ full, const uint16_t* __restrict__ src, int64_t half,
                                   int64_t total, int64_t off, int64_t n_total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    const int64_t t = i / total, c = i - t * total;
    full[i] = (c >= off && c < off + half) ? src[t * half + (c - off)] : (uint16_t) 0;
}

__global__ void tp_widen_kernel(float* __restrict__ out, const uint16_t* __restrict__ row, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(__ushort_as_bfloat16(row[i]));
}

}  // namespace

void tp_place_bf16_f32(uint16_t* full, const float* src, int64_t half, int64_t total, int64_t off, int n,
                       void* stream) {
    const int64_t n_total = (int64_t) n * total;
    tp_place_f32_kernel<<<(unsigned) ((n_total + kTpThreads - 1) / kTpThreads), kTpThreads, 0,
                          (cudaStream_t) stream>>>(full, src, half, total, off, n_total);
}

void tp_place_bf16_bf16(uint16_t* full, const uint16_t* src, int64_t half, int64_t total, int64_t off, int n,
                        void* stream) {
    const int64_t n_total = (int64_t) n * total;
    tp_place_bf_kernel<<<(unsigned) ((n_total + kTpThreads - 1) / kTpThreads), kTpThreads, 0,
                         (cudaStream_t) stream>>>(full, src, half, total, off, n_total);
}

void tp_bf16_widen(float* out, const uint16_t* row, int64_t n, void* stream) {
    tp_widen_kernel<<<(unsigned) ((n + kTpThreads - 1) / kTpThreads), kTpThreads, 0, (cudaStream_t) stream>>>(
        out, row, n);
}

}  // namespace basalt::kernels
