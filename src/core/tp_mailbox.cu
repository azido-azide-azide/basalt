// src/core/tp_mailbox.cu - the two-card warp-epoch exchange (see include/basalt/core/tp_mailbox.hpp).
#include <cstdlib>
#include "basalt/core/tp_mailbox.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/log.hpp"

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

// The grid cap: exchange blocks per SM (0: uncapped, the pre-11-24 geometry where a whole-chunk exchange is
// ~10,000 blocks). Read once (the first open), so every launch of the process - captures included - agrees.
int g_grid_bpsm = -1;
int g_sm_count[2] = {0, 0};

int grid_bpsm() {
    if (g_grid_bpsm < 0) {
        const char* e = std::getenv("BASALT_TP_GRID_BPSM");
        g_grid_bpsm = (e != nullptr && *e != '\0') ? std::atoi(e) : 4;
        if (g_grid_bpsm < 0) g_grid_bpsm = 0;
    }
    return g_grid_bpsm;
}

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

// This card's re-read of its OWN published slot bytes. The peer's writes are never involved here, so only the
// card's own store->load order matters: the VMM payload wants `__ldcv` (bypass any L1 line, hit L2 where the
// card's own `__stwt` left the data); the pinned payload must NOT be a volatile read - a PCIe read can pass
// this card's own posted writes to host memory, while its L2 always holds the freshest copy of what it wrote.
__device__ __forceinline__ Vec self_slot_read(const Vec* p, bool wt) { return wt ? __ldcv(p) : *p; }

// One lane of the protocol, pre-11-24: publish this card's vectors, release the lane's epoch, wait for the
// peer's epoch of the same round, read the peer's vectors, combine. exchange_stage() now serves several lanes
// per warp with the same per-lane steps split across its two phases; kept as the protocol's shape.
// `wt` says the payload is device memory: its publish store is then `st.global.wt`, which writes through to
// memory instead of letting the card retire the block while it is still in its own caches (a plain store from
// the writer is exactly how this rig once produced a first-token garbage bug over the pinned path).
template <class Combine>
[[maybe_unused]] __device__ __forceinline__ bool tp_lane(int rank, int warp, int lane, Vec (&mine)[kG], Vec (&peer)[kG],
                                        const int vecs, Vec* mine_payload, uint32_t* mine_flags,
                                        const Vec* peer_payload, const uint32_t* peer_flags,
                                        uint32_t* epochs, uint32_t* reads, uint32_t* hang, bool wt, Combine combine) {
    const int base = warp * (32 * kG) + lane;
#pragma unroll
    for (int j = 0; j < kG; ++j) {
        const int v = base + j * 32;
        if (v < vecs) mine[j] = combine.load_mine(v);
    }
#pragma unroll
    for (int j = 0; j < kG; ++j) {
        const int v = base + j * 32;
        if (v < vecs) {
            if (wt) __stwt(&mine_payload[v], mine[j]);
            else mine_payload[v] = mine[j];
        }
    }
    __syncwarp();
    const uint32_t target = epochs[warp] + 1u;   // the round this lane publishes and waits, whatever the lane count
    int ok = 1;
    if (lane == 0) {
        flag_release(&mine_flags[warp * kFlagStride], target);
        unsigned spins = 0;
        while (static_cast<int32_t>(flag_acquire(&peer_flags[warp * kFlagStride]) - target) < 0) {
            __nanosleep(kPollSleepNs);
            if (warp == 0 && (spins == 2000000u || spins == 10000000u))
                printf("mb: rank %d warp 0 waits epoch %u, peer flag %u, spins %u\n", rank, target,
                       flag_acquire(&peer_flags[warp * kFlagStride]), spins);
            if (++spins % 1024u == 0u && (spins > kSpinLimit || flag_acquire(hang) != 0u)) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*hang)
                    .store(1u, cuda::memory_order_relaxed);
                ok = 0;
                break;
            }
        }
        epochs[warp] = target;  // the epoch advances whatever happens: rounds stay aligned with the peer's
        flag_release(&reads[warp], target);   // and this lane has finished reading this round
    }
    __syncwarp();
    if (__shfl_sync(0xffffffffu, ok, 0) == 0) return false;
#pragma unroll
    for (int j = 0; j < kG; ++j) {
        const int v = base + j * 32;
        if (v < vecs) peer[j] = __ldcv(&peer_payload[v]);
    }
    combine.finish(base, mine, peer, vecs);
    // The split form can queue a slot's next round before the earlier finish(), so the publish side waits the
    // peer's read epoch explicitly (see exchange_add_bf16_publish_kernel).  The one-call form keeps its own
    // ordering: this card's round n + 2 begins only after its round n + 1 waited the peer's flag n + 1, which the
    // peer released after reading round n.
    return true;
}

// The whole kernel body. One-call form, bounded grid: a warp of a capped grid serves several logical lanes
// (stride of the grid's warp count) in TWO phases - it publishes and releases every lane it owns first, then
// waits, reads and combines them one by one, re-reading its own published vectors from the slot (the same bits
// the register held: publish writes them verbatim). The cap exists because the whole-chunk geometry is ~10,000
// blocks of 64 threads: blocks past the resident wave can neither publish (the peer's pull of the previous
// wave's data stalls behind them) nor run beside the compute they are meant to overlap with - and every one of
// them adds its own poll of a host-pinned flag line. Measured at full grid, the spinners alone saturate the
// host link (see dense-tp-port: `BASALT_TP_GRID_BPSM`). Deadlock-free by construction: both cards' warps walk
// their lanes in the same order and every wait targets a flag the peer's publish phase releases unconditionally.
template <class Combine>
__device__ __forceinline__ void exchange_stage(int rank, Vec* mine_payload, uint32_t* mine_flags,
                                               const Vec* peer_payload, const uint32_t* peer_flags,
                                               uint32_t* epochs, uint32_t* reads, uint32_t* hang, const int vecs, bool wt,
                                               Combine combine) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int nwarp = static_cast<int>((gridDim.x * blockDim.x) >> 5);
    const int unit = 32 * kG;
    const int ngroups = (vecs + unit - 1) / unit;
    for (int g = warp; g < ngroups; g += nwarp) {
        const int base = g * unit + lane;
#pragma unroll
        for (int j = 0; j < kG; ++j) {
            const int v = base + j * 32;
            if (v < vecs) {
                const Vec x = combine.load_mine(v);
                if (wt) __stwt(&mine_payload[v], x);
                else mine_payload[v] = x;
            }
        }
        __syncwarp();
        if (lane == 0) {
            const uint32_t target = epochs[g] + 1u;
            flag_release(&mine_flags[g * kFlagStride], target);
            epochs[g] = target;   // advance whatever happens after: rounds stay aligned with the peer's
        }
    }
    for (int g = warp; g < ngroups; g += nwarp) {
        const int base = g * unit + lane;
        int ok = 1;
        if (lane == 0) {
            const uint32_t target = epochs[g];   // the round phase 1 published (this lane wrote it: own-store own-load)
            unsigned spins = 0;
            while (static_cast<int32_t>(flag_acquire(&peer_flags[g * kFlagStride]) - target) < 0) {
                __nanosleep(kPollSleepNs);
                if (g == 0 && warp == 0 && (spins == 2000000u || spins == 10000000u))
                    printf("mb: rank %d lane 0 waits epoch %u, peer flag %u, spins %u\n", rank, target,
                           flag_acquire(&peer_flags[g * kFlagStride]), spins);
                if (++spins % 1024u == 0u && (spins > kSpinLimit || flag_acquire(hang) != 0u)) {
                    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*hang)
                        .store(1u, cuda::memory_order_relaxed);
                    ok = 0;
                    break;
                }
            }
            flag_release(&reads[g], target);   // as in tp_lane: this lane's round is spent reading (its own re-read below is slot-stable)
        }
        __syncwarp();
        if (__shfl_sync(0xffffffffu, ok, 0) == 0) return;
        Vec mine[kG] = {}, peer[kG] = {};
#pragma unroll
        for (int j = 0; j < kG; ++j) {
            const int v = base + j * 32;
            if (v < vecs) {
                peer[j] = __ldcv(&peer_payload[v]);
                mine[j] = self_slot_read(&mine_payload[v], wt);
            }
        }
        combine.finish(base, mine, peer, vecs);
    }
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

// Enqueue-time agreement: the pair's cards must say the same geometry every round - the same value count AND
// the same exchange kind, since the kinds' warp grids differ at equal counts (bf16/add: n/8 vectors, f32: n/4).
// Rank 1's call for a round comes first (the drivers queue the deep card's step ahead); rank 0's compares.
// Zero: off (a reopen clears it). The kind rides above the count: no payload reaches kKindUnit values.
constexpr size_t kKindUnit = size_t(1) << 48;
const char* kind_name(size_t kind) { return kind == 0 ? "bf16" : kind == 1 ? "add" : "f32"; }
bool n_agree(size_t* last, size_t slot, int rank, size_t n, size_t kind, std::string& err) {
    const size_t key = n + kind * kKindUnit;
    if (rank != 0) { last[slot] = key; return true; }
    if (last[slot] != 0 && last[slot] != key) {
        err = "tp mailbox: slot " + std::to_string(slot) + "'s round carries " + std::to_string(last[slot] % kKindUnit) +
              " values (" + kind_name(last[slot] / kKindUnit) + ") on card 1 and " + std::to_string(n) + " values (" +
              kind_name(kind) + ") on card 0 (the cards disagree; the pair will not turn)";
        return false;
    }
    return true;
}

__global__ __launch_bounds__(kBlockThreads) void exchange_bf16_kernel(
    VecBf16* partial, Vec* mine_payload, uint32_t* mine_flags, const Vec* peer_payload,
    const uint32_t* peer_flags, uint32_t* epochs, uint32_t* reads, uint32_t* hang, int vecs, int rank, bool wt) {
    exchange_stage(rank, mine_payload, mine_flags, peer_payload, peer_flags, epochs, reads, hang, vecs, wt,
                   SumBf16{partial});
}

__global__ __launch_bounds__(kBlockThreads) void exchange_add_bf16_kernel(
    const float* mine, VecBf16* base, Vec* mine_payload, uint32_t* mine_flags,
    const Vec* peer_payload, const uint32_t* peer_flags, uint32_t* epochs, uint32_t* reads, uint32_t* hang, int vecs,
    int rank, bool wt) {
    exchange_stage(rank, mine_payload, mine_flags, peer_payload, peer_flags, epochs, reads, hang, vecs, wt,
                   AddBf16{mine, base});
}

__global__ __launch_bounds__(kBlockThreads) void exchange_f32_kernel(
    Vec* partial, uint2* out16, Vec* mine_payload, uint32_t* mine_flags, const Vec* peer_payload,
    const uint32_t* peer_flags, uint32_t* epochs, uint32_t* reads, uint32_t* hang, int vecs, int rank, bool wt) {
    exchange_stage(rank, mine_payload, mine_flags, peer_payload, peer_flags, epochs, reads, hang, vecs, wt,
                   SumF32{partial, out16});
}

// Publish this card's BF16 rounding of its fp32 product and release the lane's epoch: the first half of the split
// exchange.  Nothing waits here, so the caller's next kernels are queued behind this one on its stream but the
// peer's payload is not needed yet.
__global__ __launch_bounds__(kBlockThreads) void exchange_add_bf16_publish_kernel(
    const float* mine, Vec* mine_payload, uint32_t* mine_flags, const uint32_t* peer_reads, uint32_t* epochs,
    uint32_t* hang, int vecs, int rank, bool wt) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int nwarp = static_cast<int>((gridDim.x * blockDim.x) >> 5);
    const int unit = 32 * kG;
    const int ngroups = (vecs + unit - 1) / unit;
    for (int g = warp; g < ngroups; g += nwarp) {
    const int base = g * unit + lane;
    Vec pub[kG] = {};
#pragma unroll
    for (int j = 0; j < kG; ++j) {
        const int v = base + j * 32;
        if (v < vecs) pub[j] = AddBf16{mine, nullptr}.load_mine(v);
    }
#pragma unroll
    for (int j = 0; j < kG; ++j) {
        const int v = base + j * 32;
        if (v < vecs) {
            if (wt) __stwt(&mine_payload[v], pub[j]);
            else mine_payload[v] = pub[j];
        }
    }
    __syncwarp();
    if (lane == 0) {
        const uint32_t target = epochs[g] + 1u;
        // The slot alternates, so this round reuses the payload two rounds back: wait until the peer has read that
        // round, or this card would overwrite values it has not consumed yet.  In a lockstep pair this is already
        // true; it is what lets begin() be queued ahead of the earlier finish().
        unsigned spins = 0;
        while (static_cast<int32_t>(flag_acquire(&peer_reads[g]) - (target - 1u)) < 0) {
            __nanosleep(kPollSleepNs);
            if (++spins % 1024u == 0u && (spins > kSpinLimit || flag_acquire(hang) != 0u)) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*hang).store(1u, cuda::memory_order_relaxed);
                break;
            }
        }
        flag_release(&mine_flags[g * kFlagStride], target);
        epochs[g] = target;
    }
    __syncwarp();
    }
}

// Wait the peer's epoch of this lane's round, then combine the two published payloads into `base`: the second half
// of the split exchange.  This card's own published values are re-read from its slot (the registers are gone), so
// the bits are exactly the ones `begin` published.
__global__ __launch_bounds__(kBlockThreads) void exchange_add_bf16_combine_kernel(
    VecBf16* base, Vec* mine_payload, uint32_t* mine_flags, const Vec* peer_payload,
    const uint32_t* peer_flags, uint32_t* epochs, uint32_t* reads, uint32_t* hang, int vecs, int rank, bool wt) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int nwarp = static_cast<int>((gridDim.x * blockDim.x) >> 5);
    const int unit = 32 * kG;
    const int ngroups = (vecs + unit - 1) / unit;
    for (int g = warp; g < ngroups; g += nwarp) {
    const int base_i = g * unit + lane;
    const uint32_t target = epochs[g];   // the round this lane published
    if (lane == 0) {
        unsigned spins = 0;
        while (static_cast<int32_t>(flag_acquire(&peer_flags[g * kFlagStride]) - target) < 0) {
            __nanosleep(kPollSleepNs);
            if (++spins % 1024u == 0u && (spins > kSpinLimit || flag_acquire(hang) != 0u)) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(*hang).store(1u, cuda::memory_order_relaxed);
                break;
            }
        }
    }
    __syncwarp();
    if (flag_acquire(hang) != 0u) return;
    Vec mine[kG] = {}, peer[kG] = {};
#pragma unroll
    for (int j = 0; j < kG; ++j) {
        const int v = base_i + j * 32;
        if (v < vecs) {
            mine[j] = self_slot_read(&mine_payload[v], wt);
            peer[j] = __ldcv(&peer_payload[v]);
        }
    }
    AddBf16{nullptr, base}.finish(base_i, mine, peer, vecs);
    __syncwarp();
    if (lane == 0) flag_release(&reads[g], target);   // this lane's payload is consumed
    }
}
dim3 exchange_grid(size_t vecs, int rank) {
    const size_t warps = (vecs + static_cast<size_t>(32 * kG) - 1) / (32 * kG);
    size_t blocks = (warps + kBlockThreads / 32 - 1) / (kBlockThreads / 32);
    // the SM cap (BASALT_TP_GRID_BPSM, 0: uncapped - the pre-11-24 geometry): the exchange kernel's blocks must
    // not occupy the whole machine - beside compute (the overlap's premise) and not all at once polling host
    // flags. The kernels' lane loops grid-stride, so a capped grid moves the same bytes in the same order.
    const int bpsm = grid_bpsm();
    if (bpsm > 0) {
        const size_t cap = static_cast<size_t>(g_sm_count[rank < 2 ? rank : 0]) * static_cast<size_t>(bpsm);
        if (cap > 0 && blocks > cap) blocks = cap;
    }
    return dim3(static_cast<unsigned>(blocks));
}

}  // namespace

// One slot: its release lines in the shared pinned block (the same mapped addresses for both cards) and its payload
// region per rank - in the pinned block, or in that rank's device VMM range when `dev_payload_`.
struct TpMailbox::Slot {
    uint32_t* flags[2] = {nullptr, nullptr};
    Vec* payload[2] = {nullptr, nullptr};
    uint32_t* epochs[2] = {nullptr, nullptr};
    uint32_t* reads[2] = {nullptr, nullptr};   // the round whose payload this card has finished reading
};

TpMailbox::~TpMailbox() { close(); }

// One warp, one block: wait for the driver's release count to reach this step's token. A nanosleep spin on a
// host word - one SM, no timeout: the stall-watchdog is the safety net, and the point is precisely that this
// wait must survive however long the host takes to finish queueing the pair.
__global__ void tp_gate_kernel(volatile uint32_t* word, uint32_t token) {
    while (*word < token) __nanosleep(200);
}

uint32_t TpMailbox::gate_raise() { return ++gate_seq_; }
void TpMailbox::gate_release() {
    if (gate_) __atomic_store_n(gate_, gate_seq_, __ATOMIC_RELEASE);
}
void TpMailbox::post_gate(uint32_t token, void* stream) {
    if (!gate_) return;
    tp_gate_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<volatile uint32_t*>(gate_), token);
}

// The driver's divergence mark: an exchange that failed to enqueue leaves the cards' rounds unpaired, so the
// pair's results have split before any kernel spins. Sticky, as a kernel's timeout is (a reopen clears it).
void TpMailbox::poison() {
    if (hang_ != nullptr) std::atomic_ref<uint32_t>{*hang_}.store(1u, std::memory_order_relaxed);
}

bool TpMailbox::open(int rank0_dev, int rank1_dev, size_t slots, size_t max_bytes, std::string& err,
                     bool want_device) {
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
    // The device path's pinned block holds every (slot, rank) pair's release lines in their own region, each
    // `line_bytes` (one 64-byte line per warp lane); the fallback keeps the original interleaved layout.
    const size_t dev_line_total = line_bytes_ * 2 * slots;

    // ---- the payload region: device VMM when the driver offers it, else the pinned block ----
    // One reserved range per rank holding all its slots' payloads, backed by one allocation on that rank's card
    // and mapped there.  The owner then GRANTS the peer read/write access to the same address (cuMemSetAccess),
    // so rank r publishes at its own range's address and the peer reads that same address: one mapping, one
    // traversal of the link per crossing, and no reliance on the driver's peer access.  (Mapping the handle a
    // second time into the peer's context is refused - a VA belongs to the context that reserved it.)
    dev_payload_ = false;
    if (want_device && vmm::supported()) {
        const uint64_t g = vmm::granularity();
        if (g != 0) {
            // the payload needs no alignment pad (slot_bytes_ is whole KiB); the BACKING must be a whole granule
            const uint64_t want = (uint64_t) slot_bytes_ * slots;
            const uint64_t span = (want + g - 1) / g * g;
            bool ok = true;
            // A failure at rank r must leave NOTHING behind: the ranges already made are torn down before the
            // fallback, or a later close() would unmap a VA that was never mapped (the engine's 80 MiB/rank
            // allocation does not fit a card that already holds its weights, so this path is real).
            auto tear_down = [&]() {
                for (int r = 0; r < 2; ++r) {
                    const OnDevice on(r == 0 ? rank0_dev : rank1_dev);
                    if (payload_va_[r] == nullptr) continue;
                    vmm::unmap_only(payload_va_[r], payload_span_);
                    if (payload_handle_[r] != 0) vmm::release(payload_handle_[r]);
                    vmm::release_range(payload_va_[r], payload_span_);
                    payload_va_[r] = nullptr;
                    payload_handle_[r] = 0;
                }
            };
            for (int rank = 0; rank < 2 && ok; ++rank) {
                const OnDevice on(rank == 0 ? rank0_dev : rank1_dev);
                uint8_t* va = vmm::reserve(span);
                if (va == nullptr) { ok = false; break; }
                const unsigned long long h = vmm::create(span);
                if (h == 0) { vmm::release_range(va, span); ok = false; break; }
                if (!vmm::map_handle(va, span, h)) {
                    vmm::release(h);
                    vmm::release_range(va, span);
                    ok = false;
                    break;
                }
                // the peer card reaches the same bytes at the same address
                if (!vmm::grant_access(va, span, rank == 0 ? rank1_dev : rank0_dev)) {
                    vmm::unmap_only(va, span);
                    vmm::release(h);
                    vmm::release_range(va, span);
                    ok = false;
                    break;
                }
                payload_va_[rank] = va;
                payload_handle_[rank] = h;
                payload_span_ = span;
            }
            if (ok) {
                dev_payload_ = true;
            } else {
                tear_down();
                payload_span_ = 0;
            }
        }
    }

    void* pinned = nullptr;
    // on the device path the pinned block carries only the release lines and the hang word
    const size_t pinned_total = dev_payload_ ? dev_line_total + 256 : total;
    if (cudaHostAlloc(&pinned, pinned_total, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        const cudaError_t e = cudaGetLastError();
        err = "tp mailbox: out of page-locked memory for " + std::to_string(pinned_total >> 20) + " MiB (" +
              std::string(cudaGetErrorString(e)) + ")";
        return false;
    }
    std::memset(pinned, 0, pinned_total);
    block_ = static_cast<uint8_t*>(pinned);
    hang_ = reinterpret_cast<uint32_t*>(block_ + pinned_total - 128);
    gate_ = reinterpret_cast<uint32_t*>(block_ + pinned_total - 64);
    gate_seq_ = 0;
    *gate_ = 0;
    slot_views_ = new Slot[slots];
    n_slots_ = slots;
    dev0_ = rank0_dev;
    dev1_ = rank1_dev;
    // the grid cap's per-rank SM counts (queried once; a failed query leaves 0, which the cap check treats as
    // uncapped for that rank)
    for (int rank = 0; rank < 2; ++rank) {
        const OnDevice on(rank == 0 ? rank0_dev : rank1_dev);
        int sm = 0;
        if (cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, rank == 0 ? rank0_dev : rank1_dev) ==
            cudaSuccess)
            g_sm_count[rank] = sm;
    }
    // The read-done epochs live in mapped host memory, like the release lines: the split form makes a card wait
    // for the PEER's read epoch, so they cannot sit in one card's device memory the way the publish epochs do.
    {
        const OnDevice on(rank0_dev);
        if (cudaHostAlloc(reinterpret_cast<void**>(&reads_), slots * 2 * lanes * sizeof(uint32_t),
                          cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            reads_ == nullptr) {
            (void)cudaGetLastError();
            err = "tp mailbox: out of page-locked memory for the read epochs";
            close();
            return false;
        }
        std::memset(reads_, 0, slots * 2 * lanes * sizeof(uint32_t));
    }
    for (size_t s = 0; s < slots; ++s) {
        // the release lines keep their per-slot, per-rank layout on the fallback path; on the device path each
        // (slot, rank) pair's lines have a region of their own, and the payload lives in that rank's device range
        for (int rank = 0; rank < 2; ++rank) {
            const size_t lines_at = dev_payload_ ? (s * 2 + rank) * line_bytes_ : slot_span * s + rank * rank_span;
            slot_views_[s].flags[rank] = reinterpret_cast<uint32_t*>(block_ + lines_at);
            slot_views_[s].reads[rank] = reads_ + (s * 2 + rank) * lanes;
            uint8_t* pay = dev_payload_ ? payload_va_[rank] + s * slot_bytes_ : block_ + lines_at + line_bytes_;
            slot_views_[s].payload[rank] = reinterpret_cast<Vec*>(pay);
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
            for (size_t s = 0; s < n_slots_; ++s) {
                if (slot_views_[s].epochs[rank] != nullptr) cudaFree(slot_views_[s].epochs[rank]);
            }
        }
        delete[] slot_views_;
        slot_views_ = nullptr;
    }
    if (block_ != nullptr) {
        cudaFreeHost(block_);
        block_ = nullptr;
    }
    if (reads_ != nullptr) {
        cudaFreeHost(reads_);
        reads_ = nullptr;
    }
    if (dev_payload_) {
        for (int rank = 0; rank < 2; ++rank) {
            if (payload_va_[rank] != nullptr) {
                for (int d = 0; d < 2; ++d) {
                    const int dev = d == 0 ? dev0_ : dev1_;
                    if (dev < 0) continue;
                    const OnDevice on(dev);
                    vmm::unmap_only(payload_va_[rank], payload_span_);
                }
            }
            if (payload_handle_[rank] != 0) vmm::release(payload_handle_[rank]);
            if (payload_va_[rank] != nullptr) {
                const OnDevice on(rank == 0 ? dev0_ : dev1_);
                vmm::release_range(payload_va_[rank], payload_span_);
            }
            payload_va_[rank] = nullptr;
            payload_handle_[rank] = 0;
        }
        payload_span_ = 0;
        dev_payload_ = false;
    }
    hang_ = nullptr;
    n_slots_ = 0;
    slot_bytes_ = line_bytes_ = 0;
    dev0_ = dev1_ = -1;
    last_n_.clear();   // a reopen starts the agreement check off, as its comment says
}

bool TpMailbox::exchange_bf16(int rank, size_t slot, void* partial, size_t bytes, void* stream,
                              std::string& err) {
    if (bytes == 0 || bytes % kVecBytes != 0 || slot >= n_slots_ || bytes > slot_bytes_) {
        err = "tp mailbox: a BF16 exchange's payload is out of range";
        return false;
    }
    const size_t vecs = bytes / kVecBytes;
    if (last_n_.size() < n_slots_) last_n_.assign(n_slots_, 0);
    if (!n_agree(last_n_.data(), slot, rank, bytes / kVecBytes * 8, 0, err)) return false;
    const Slot& s = slot_views_[slot];
    const OnDevice on(rank == 0 ? dev0_ : dev1_);
    exchange_bf16_kernel<<<exchange_grid(vecs, rank), kBlockThreads, 0,
                           static_cast<cudaStream_t>(stream)>>>(
        static_cast<VecBf16*>(partial), s.payload[rank], s.flags[rank], s.payload[1 - rank],
        s.flags[1 - rank], s.epochs[rank], s.reads[rank], hang_, static_cast<int>(vecs), rank, dev_payload_);
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
    if (last_n_.size() < n_slots_) last_n_.assign(n_slots_, 0);
    if (!n_agree(last_n_.data(), slot, rank, n, 1, err)) return false;
    const Slot& s = slot_views_[slot];
    const OnDevice on(rank == 0 ? dev0_ : dev1_);
    exchange_add_bf16_kernel<<<exchange_grid(vecs, rank), kBlockThreads, 0,
                               static_cast<cudaStream_t>(stream)>>>(
        mine, reinterpret_cast<VecBf16*>(base), s.payload[rank], s.flags[rank], s.payload[1 - rank],
        s.flags[1 - rank], s.epochs[rank], s.reads[rank], hang_, static_cast<int>(vecs), rank, dev_payload_);
    if (cudaGetLastError() != cudaSuccess) {
        err = "tp mailbox: launching the add exchange failed";
        return false;
    }
    return true;
}

bool TpMailbox::exchange_add_bf16_begin(int rank, size_t slot, const float* mine, size_t n, void* stream,
                                        std::string& err) {
    if (n == 0 || n % 8 != 0) {
        err = "tp mailbox: an add exchange's value count is not a multiple of eight";
        return false;
    }
    const size_t vecs = n / 8;
    if (slot >= n_slots_ || vecs * kVecBytes > slot_bytes_) {
        err = "tp mailbox: an add exchange's payload is out of range";
        return false;
    }
    if (last_n_.size() < n_slots_) last_n_.assign(n_slots_, 0);
    if (!n_agree(last_n_.data(), slot, rank, n, 1, err)) return false;
    const Slot& s = slot_views_[slot];
    const OnDevice on(rank == 0 ? dev0_ : dev1_);
    exchange_add_bf16_publish_kernel<<<exchange_grid(vecs, rank), kBlockThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        mine, s.payload[rank], s.flags[rank], s.reads[1 - rank], s.epochs[rank], hang_, static_cast<int>(vecs), rank,
        dev_payload_);
    if (cudaGetLastError() != cudaSuccess) {
        err = "tp mailbox: launching the add exchange's publish failed";
        return false;
    }
    return true;
}

bool TpMailbox::exchange_add_bf16_finish(int rank, size_t slot, uint16_t* base, size_t n, void* stream,
                                         std::string& err) {
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
    exchange_add_bf16_combine_kernel<<<exchange_grid(vecs, rank), kBlockThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<VecBf16*>(base), s.payload[rank], s.flags[rank], s.payload[1 - rank],
        s.flags[1 - rank], s.epochs[rank], s.reads[rank], hang_, static_cast<int>(vecs), rank, dev_payload_);
    if (cudaGetLastError() != cudaSuccess) {
        err = "tp mailbox: launching the add exchange's combine failed";
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
    if (last_n_.size() < n_slots_) last_n_.assign(n_slots_, 0);
    if (!n_agree(last_n_.data(), slot, rank, n, 2, err)) return false;
    const Slot& s = slot_views_[slot];
    const OnDevice on(rank == 0 ? dev0_ : dev1_);
    exchange_f32_kernel<<<exchange_grid(vecs, rank), kBlockThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<Vec*>(partial), reinterpret_cast<uint2*>(bf16_out), s.payload[rank],
        s.flags[rank], s.payload[1 - rank], s.flags[1 - rank], s.epochs[rank], s.reads[rank], hang_,
        static_cast<int>(vecs), rank, dev_payload_);
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