// src/core/tp_mailbox.cu - the two-card warp-epoch exchange (see include/basalt/core/tp_mailbox.hpp).
#include "basalt/core/tp_mailbox.hpp"
#include "basalt/core/on_device.hpp"

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstring>

namespace basalt::core {
namespace {

// One probe sleeps this long; the spin limit below bounds a warp's wait for its peer's epoch. The budget must
// cover the pair's cold start: the first steps pay lazy module loading on each card separately (CUDA keeps it
// per-process, but the cards' first touches land seconds apart under capture and instantiate queues), and a
// legitimately late peer must never be read as a divergence. A truly diverged pair burns this before hang_reported.
constexpr unsigned kPollSleepNs = 100;
constexpr unsigned kSpinLimit = 20000000u;
// Words between two warps' release lines: one host cache line per lane.
constexpr int kFlagStride = 16;
// Threads per exchange block: two warps, each warp an independent lane of the protocol.
constexpr int kBlockThreads = 64;
// Vectors per lane (the geometry every exchange of this mailbox uses).
constexpr int kG = TpMailbox::kVecsPerLane;

using Vec = uint4;  // 16 bytes

union VecBf16 {
    Vec raw;
    __nv_bfloat162 pair[4];
};

__device__ __forceinline__ uint32_t flag_acquire(const uint32_t* p) {
    return cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*const_cast<uint32_t*>(p))
        .load(cuda::memory_order_acquire);
}

__device__ __forceinline__ void flag_release(uint32_t* p, uint32_t v) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*p)
        .store(v, cuda::memory_order_release);
}

// One lane of the protocol: publish this card's vectors, release the lane's epoch, wait for the peer's epoch
// of the same round, then read the peer's vectors on the non-coherent path (the peer writes them from its own
// card while this kernel runs) and hand everything to `combine`. All vector loads sit in flight before the
// stores. Returns false, warp-uniformly, once the wait gave up; the hang word is set for every warp then.
template <class Combine>
__device__ __forceinline__ bool tp_lane(int warp, int lane, Vec (&mine)[kG], Vec (&peer)[kG],
                                        const int vecs, Vec* mine_payload, uint32_t* mine_flags,
                                        const Vec* peer_payload, const uint32_t* peer_flags,
                                        uint32_t* epochs, uint32_t* hang, Combine combine) {
    const int base = warp * (32 * kG) + lane;
#pragma unroll
    for (int j = 0; j < kG; ++j) {
        const int v = base + j * 32;
        if (v < vecs) mine[j] = combine.load_mine(v);
    }
#pragma unroll
    for (int j = 0; j < kG; ++j) {
        const int v = base + j * 32;
        if (v < vecs) mine_payload[v] = mine[j];
    }
    __syncwarp();
    int ok = 1;
    if (lane == 0) {
        const uint32_t target = epochs[warp] + 1u;
        flag_release(&mine_flags[warp * kFlagStride], target);
        unsigned spins = 0;
        while (static_cast<int32_t>(flag_acquire(&peer_flags[warp * kFlagStride]) - target) < 0) {
            __nanosleep(kPollSleepNs);
            if (++spins % 1024u == 0u && (spins > kSpinLimit || flag_acquire(hang) != 0u)) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*hang)
                    .store(1u, cuda::memory_order_relaxed);
                ok = 0;
                break;
            }
        }
        epochs[warp] = target;  // the epoch advances whatever happens: rounds stay aligned with the peer's
    }
    __syncwarp();
    if (__shfl_sync(0xffffffffu, ok, 0) == 0) return false;
#pragma unroll
    for (int j = 0; j < kG; ++j) {
        const int v = base + j * 32;
        if (v < vecs) peer[j] = __ldcv(&peer_payload[v]);
    }
    combine.finish(base, mine, peer, vecs);
    return true;
}

// The whole kernel body: one lane per warp of the grid, the grid's last warps idle past the payload.
template <class Combine>
__device__ __forceinline__ void exchange_stage(Vec* mine_payload, uint32_t* mine_flags,
                                               const Vec* peer_payload, const uint32_t* peer_flags,
                                               uint32_t* epochs, uint32_t* hang, const int vecs,
                                               Combine combine) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (warp * (32 * kG) >= vecs) return;
    Vec mine[kG] = {}, peer[kG] = {};
    tp_lane(warp, lane, mine, peer, vecs, mine_payload, mine_flags, peer_payload, peer_flags,
            epochs, hang, combine);
}

// Publish this card's BF16 partial as-is; combine widens both values, adds and rounds once back into the
// partial, so both cards keep equal bits.
struct SumBf16 {
    VecBf16* partial;
    __device__ inline Vec load_mine(int v) { return partial[v].raw; }
    __device__ inline void finish(int base, Vec (&mine)[kG], Vec (&peer)[kG], const int vecs) {
#pragma unroll
        for (int j = 0; j < kG; ++j) {
            const int v = base + j * 32;
            if (v >= vecs) continue;
            VecBf16 a, b, s;
            a.raw = mine[j];
            b.raw = peer[j];
#pragma unroll
            for (int p = 0; p < 4; ++p)
                s.pair[p] = __floats2bfloat162_rn(__low2float(a.pair[p]) + __low2float(b.pair[p]),
                                                  __high2float(a.pair[p]) + __high2float(b.pair[p]));
            partial[v].raw = s.raw;
        }
    }
};

// Publish the BF16 rounding of this card's fp32 product; combine adds the two rounded products and the
// residual row in fp32 and rounds the sum once into the residual. A vector carries eight values.
struct AddBf16 {
    const float* mine;
    VecBf16* base;
    __device__ inline Vec load_mine(int v) {
        const float4* f = reinterpret_cast<const float4*>(mine);
        const float4 lo = f[2 * v], hi = f[2 * v + 1];
        VecBf16 s;
        s.pair[0] = __floats2bfloat162_rn(lo.x, lo.y);
        s.pair[1] = __floats2bfloat162_rn(lo.z, lo.w);
        s.pair[2] = __floats2bfloat162_rn(hi.x, hi.y);
        s.pair[3] = __floats2bfloat162_rn(hi.z, hi.w);
        return s.raw;
    }
    __device__ inline void finish(int base_i, Vec (&mine)[kG], Vec (&peer)[kG], const int vecs) {
#pragma unroll
        for (int j = 0; j < kG; ++j) {
            const int v = base_i + j * 32;
            if (v >= vecs) continue;
            VecBf16 a, b, x, s;
            a.raw = mine[j];
            b.raw = peer[j];
            x.raw = base[v].raw;
#pragma unroll
            for (int p = 0; p < 4; ++p)
                s.pair[p] = __floats2bfloat162_rn(
                    __low2float(x.pair[p]) + __low2float(a.pair[p]) + __low2float(b.pair[p]),
                    __high2float(x.pair[p]) + __high2float(a.pair[p]) + __high2float(b.pair[p]));
            base[v].raw = s.raw;
        }
    }
};

// Publish the fp32 partial as-is; combine adds in fp32 (commutative, so the cards' bits agree), replaces the
// partial and optionally stores the sum rounded once to BF16. A vector carries four values.
struct SumF32 {
    Vec* partial;
    uint2* out16;
    __device__ inline Vec load_mine(int v) { return partial[v]; }
    __device__ inline void finish(int base, Vec (&mine)[kG], Vec (&peer)[kG], const int vecs) {
#pragma unroll
        for (int j = 0; j < kG; ++j) {
            const int v = base + j * 32;
            if (v >= vecs) continue;
            const float4* a = reinterpret_cast<const float4*>(&mine[j]);
            const float4* b = reinterpret_cast<const float4*>(&peer[j]);
            float4 s;
            s.x = __fadd_rn(a->x, b->x);
            s.y = __fadd_rn(a->y, b->y);
            s.z = __fadd_rn(a->z, b->z);
            s.w = __fadd_rn(a->w, b->w);
            partial[v] = *reinterpret_cast<Vec*>(&s);
            if (out16 != nullptr) {
                const __nv_bfloat162 lo = __floats2bfloat162_rn(s.x, s.y);
                const __nv_bfloat162 hi = __floats2bfloat162_rn(s.z, s.w);
                uint2 packed;
                packed.x = *reinterpret_cast<const uint32_t*>(&lo);
                packed.y = *reinterpret_cast<const uint32_t*>(&hi);
                out16[v] = packed;
            }
        }
    }
};

__global__ __launch_bounds__(kBlockThreads) void exchange_bf16_kernel(
    VecBf16* partial, Vec* mine_payload, uint32_t* mine_flags, const Vec* peer_payload,
    const uint32_t* peer_flags, uint32_t* epochs, uint32_t* hang, int vecs) {
    exchange_stage(mine_payload, mine_flags, peer_payload, peer_flags, epochs, hang, vecs,
                   SumBf16{partial});
}

__global__ __launch_bounds__(kBlockThreads) void exchange_add_bf16_kernel(
    const float* mine, VecBf16* base, Vec* mine_payload, uint32_t* mine_flags,
    const Vec* peer_payload, const uint32_t* peer_flags, uint32_t* epochs, uint32_t* hang, int vecs) {
    exchange_stage(mine_payload, mine_flags, peer_payload, peer_flags, epochs, hang, vecs,
                   AddBf16{mine, base});
}

__global__ __launch_bounds__(kBlockThreads) void exchange_f32_kernel(
    Vec* partial, uint2* out16, Vec* mine_payload, uint32_t* mine_flags, const Vec* peer_payload,
    const uint32_t* peer_flags, uint32_t* epochs, uint32_t* hang, int vecs) {
    exchange_stage(mine_payload, mine_flags, peer_payload, peer_flags, epochs, hang, vecs,
                   SumF32{partial, out16});
}

dim3 exchange_grid(size_t vecs) {
    const size_t warps = (vecs + static_cast<size_t>(32 * kG) - 1) / (32 * kG);
    const size_t blocks = (warps + kBlockThreads / 32 - 1) / (kBlockThreads / 32);
    return dim3(static_cast<unsigned>(blocks));
}

}  // namespace

// One slot: its release lines and payload region in the shared pinned block (the same mapped addresses for
// both cards) and the per-lane epochs in each card's memory.
struct TpMailbox::Slot {
    uint32_t* flags[2] = {nullptr, nullptr};
    Vec* payload[2] = {nullptr, nullptr};
    uint32_t* epochs[2] = {nullptr, nullptr};
};

TpMailbox::~TpMailbox() { close(); }

bool TpMailbox::open(int rank0_dev, int rank1_dev, size_t slots, size_t max_bytes, std::string& err) {
    close();
    if (rank0_dev == rank1_dev || slots == 0 || max_bytes == 0) {
        err = "tp mailbox: needs two cards, at least one slot and a positive payload";
        return false;
    }
    const size_t unit = kVecBytes * 32 * static_cast<size_t>(kG);
    slot_bytes_ = (max_bytes + unit - 1) / unit * unit;
    const size_t lanes = (slot_bytes_ / kVecBytes + 32 * static_cast<size_t>(kG) - 1) /
                         (32 * static_cast<size_t>(kG));
    line_bytes_ = lanes * 64;
    // Two rank sections per slot: each card publishes only into its own bytes.
    const size_t rank_span = line_bytes_ + slot_bytes_;
    const size_t slot_span = 2 * rank_span;
    const size_t total = slot_span * slots + 256;
    void* pinned = nullptr;
    if (cudaHostAlloc(&pinned, total, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void)cudaGetLastError();
        err = "tp mailbox: out of page-locked memory for " + std::to_string(total >> 20) + " MiB";
        return false;
    }
    std::memset(pinned, 0, total);
    block_ = static_cast<uint8_t*>(pinned);
    hang_ = reinterpret_cast<uint32_t*>(block_ + slot_span * slots);
    slot_views_ = new Slot[slots];
    n_slots_ = slots;
    dev0_ = rank0_dev;
    dev1_ = rank1_dev;
    for (size_t s = 0; s < slots; ++s) {
        uint8_t* at = block_ + slot_span * s;
        // Under unified addressing one mapped pointer names the bytes on both cards; rank r's
        // published bytes live in the slot's r-th section.
        for (int rank = 0; rank < 2; ++rank) {
            uint8_t* sec = at + rank * (line_bytes_ + slot_bytes_);
            slot_views_[s].flags[rank] = reinterpret_cast<uint32_t*>(sec);
            slot_views_[s].payload[rank] = reinterpret_cast<Vec*>(sec + line_bytes_);
        }
    }
    for (int rank = 0; rank < 2; ++rank) {
        const OnDevice on(rank == 0 ? rank0_dev : rank1_dev);
        for (size_t s = 0; s < slots; ++s) {
            uint32_t** e = &slot_views_[s].epochs[rank];
            if (cudaMalloc(reinterpret_cast<void**>(e), lanes * sizeof(uint32_t)) != cudaSuccess ||
                cudaMemset(*e, 0, lanes * sizeof(uint32_t)) != cudaSuccess) {
                (void)cudaGetLastError();
                err = "tp mailbox: preparing the device epochs failed";
                close();
                return false;
            }
        }
    }
    return true;
}

void TpMailbox::close() {
    if (slot_views_ != nullptr) {
        for (int rank = 0; rank < 2; ++rank) {
            const int dev = rank == 0 ? dev0_ : dev1_;
            if (dev < 0) continue;
            const OnDevice on(dev);
            for (size_t s = 0; s < n_slots_; ++s)
                if (slot_views_[s].epochs[rank] != nullptr) cudaFree(slot_views_[s].epochs[rank]);
        }
        delete[] slot_views_;
        slot_views_ = nullptr;
    }
    if (block_ != nullptr) {
        cudaFreeHost(block_);
        block_ = nullptr;
    }
    hang_ = nullptr;
    n_slots_ = 0;
    slot_bytes_ = line_bytes_ = 0;
    dev0_ = dev1_ = -1;
}

bool TpMailbox::exchange_bf16(int rank, size_t slot, void* partial, size_t bytes, void* stream,
                              std::string& err) {
    if (bytes == 0 || bytes % kVecBytes != 0 || slot >= n_slots_ || bytes > slot_bytes_) {
        err = "tp mailbox: a BF16 exchange's payload is out of range";
        return false;
    }
    const size_t vecs = bytes / kVecBytes;
    const Slot& s = slot_views_[slot];
    const OnDevice on(rank == 0 ? dev0_ : dev1_);
    exchange_bf16_kernel<<<exchange_grid(vecs), kBlockThreads, 0,
                           static_cast<cudaStream_t>(stream)>>>(
        static_cast<VecBf16*>(partial), s.payload[rank], s.flags[rank], s.payload[1 - rank],
        s.flags[1 - rank], s.epochs[rank], hang_, static_cast<int>(vecs));
    if (cudaGetLastError() != cudaSuccess) {
        err = "tp mailbox: launching the BF16 exchange failed";
        return false;
    }
    return true;
}

bool TpMailbox::exchange_add_bf16(int rank, size_t slot, const float* mine, uint16_t* base, size_t n,
                                  void* stream, std::string& err) {
    // A vector carries eight values of `mine` (fp32) and of `base` (BF16).
    if (n == 0 || n % 8 != 0) {
        err = "tp mailbox: an add exchange's value count is not a multiple of eight";
        return false;
    }
    const size_t vecs = n / 8;
    if (slot >= n_slots_ || vecs * kVecBytes > slot_bytes_) {
        err = "tp mailbox: an add exchange's payload is out of range";
        return false;
    }
    const Slot& s = slot_views_[slot];
    const OnDevice on(rank == 0 ? dev0_ : dev1_);
    exchange_add_bf16_kernel<<<exchange_grid(vecs), kBlockThreads, 0,
                               static_cast<cudaStream_t>(stream)>>>(
        mine, reinterpret_cast<VecBf16*>(base), s.payload[rank], s.flags[rank], s.payload[1 - rank],
        s.flags[1 - rank], s.epochs[rank], hang_, static_cast<int>(vecs));
    if (cudaGetLastError() != cudaSuccess) {
        err = "tp mailbox: launching the add exchange failed";
        return false;
    }
    return true;
}

bool TpMailbox::exchange_f32(int rank, size_t slot, float* partial, uint16_t* bf16_out, size_t n,
                             void* stream, std::string& err) {
    // A vector carries four fp32 values (and, for `bf16_out`, eight bytes).
    if (n == 0 || n % 4 != 0) {
        err = "tp mailbox: an F32 exchange's value count is not a multiple of four";
        return false;
    }
    const size_t vecs = n / 4;
    if (slot >= n_slots_ || vecs * kVecBytes > slot_bytes_) {
        err = "tp mailbox: an F32 exchange's payload is out of range";
        return false;
    }
    const Slot& s = slot_views_[slot];
    const OnDevice on(rank == 0 ? dev0_ : dev1_);
    exchange_f32_kernel<<<exchange_grid(vecs), kBlockThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<Vec*>(partial), reinterpret_cast<uint2*>(bf16_out), s.payload[rank],
        s.flags[rank], s.payload[1 - rank], s.flags[1 - rank], s.epochs[rank], hang_,
        static_cast<int>(vecs));
    if (cudaGetLastError() != cudaSuccess) {
        err = "tp mailbox: launching the F32 exchange failed";
        return false;
    }
    return true;
}

bool TpMailbox::hang_reported() const {
    if (hang_ == nullptr) return false;
    return std::atomic_ref<uint32_t>{*hang_}.load(std::memory_order_relaxed) != 0u;
}

}  // namespace basalt::core
