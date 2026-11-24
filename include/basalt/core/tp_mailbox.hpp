// include/basalt/core/tp_mailbox.hpp - the small-tensor exchange between two cards: tensor parallelism's collective.
//
// A mailbox is a set of page-locked, device-mapped slots shared by an ordered pair of cards. One exchange runs
// as one kernel on each card, launched for the same slot with the same geometry: the kernel publishes its card's
// payload into its own slot, releases a per-warp flag, waits for the peer's flag of the same round, reads the
// peer's payload and combines it locally. There is no host step, no event and no copy-engine work between the two
// kernels of an exchange, which is what lets exchanges live inside captured CUDA graphs: each slot's per-warp
// epochs advance in lockstep over both cards, so replays need no resets and a flag left at the previous round
// never satisfies the wait.
//
// The kernel's grid is SM-capped (`BASALT_TP_GRID_BPSM`, blocks per SM, 0: uncapped): a whole-chunk geometry
// would otherwise launch ~10,000 blocks and own the machine. Capped, each warp grid-strides several logical
// lanes; the one-call form publishes all of them first (phase 1) and waits/reads/combines after (phase 2),
// re-reading its own published vectors from the slot - the same bits, a smaller, schedulable kernel.
//
// The payloads cross PCIe as card-to-host reads of pinned memory: a card never needs to name the other card's
// memory, so this works whether or not the driver granted peer access.
//
// The payload region is device VMM memory when the driver offers it (`basalt/core/vmm.hpp`): one backing per rank,
// mapped under both cards' address ranges, so each rank's publish and the peer's read are device-to-device over the
// peer link - one traversal of it per crossing instead of the two a host bounce buffer makes.  The per-warp release
// lines and the hang word stay in the page-locked block: a release is a `thread_scope_system` atomic the two cards
// must both see, which device memory cannot give them.  A device payload publishes with `st.global.wt` and reads
// with `__ldcv` (the same non-coherent pair `tp_lane` already uses for the pinned path), so the peer's write is not
// left in this card's caches.
//
// Slot use: a rank starts a slot's round n + 2 only after its round n + 1 waited for the peer's flag n + 1, which
// the peer releases from the kernel that already read the round n payload of this slot's peer line; consecutive
// exchanges therefore alternate slots. Within a captured graph, distinct slots per exchange site plus the graphs'
// own ordering between replays satisfy the same bound.
//
// A warp whose wait exceeds its bound sets a pinned hang word and leaves; the caller reads hang_reported() after
// synchronizing the round. The word is sticky: the cards' results have diverged and the mailbox must not be used
// again (reopen it).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

class TpMailbox {
public:
    // The payload unit: 16 bytes, one PCIe transaction per access.
    static constexpr size_t kVecBytes = 16;
    // Payload vectors one warp lane publishes and reads (one KiB of payload per lane).
    static constexpr int kVecsPerLane = 4;

    TpMailbox() = default;
    ~TpMailbox();
    TpMailbox(const TpMailbox&) = delete;
    TpMailbox& operator=(const TpMailbox&) = delete;

    /// Opens `slots` slots, each holding `max_bytes`, for the ordered card pair (`rank0_dev`, `rank1_dev`).
    /// Page-locked memory is allocated for the per-warp release lines and the hang word, and mapped into both cards'
    /// address spaces; each rank's payload region is device VMM memory when the driver offers it (see the header),
    /// else the same page-locked block.  `max_bytes` is rounded up to a multiple of `kVecBytes * 32 * kVecsPerLane`
    /// (the whole-geometry unit).
    bool open(int rank0_dev, int rank1_dev, size_t slots, size_t max_bytes, std::string& err,
              bool want_device = true);
    void close();

    /// True when the payloads are device VMM memory (one peer-link traversal per crossing); false on the pinned path.
    bool device_payload() const { return dev_payload_; }

    /// Bytes of one slot's payload region and the number of slots.
    size_t slot_bytes() const { return slot_bytes_; }
    size_t slots() const { return n_slots_; }

    /// BF16 payload exchange: `partial` (device, `bytes`) is replaced in place on both cards by the sum of the
    /// two cards' partials (fp32 add of the two BF16 values, rounded once). `slot` alternates between calls.
    bool exchange_bf16(int rank, size_t slot, void* partial, size_t bytes, void* stream, std::string& err);

    /// BF16 payload exchange with the residual add fused: reads this card's fp32 product `mine` (`n` values),
    /// publishes the BF16 rounding of it, and replaces `base` (device BF16, `n` values, the residual rows) with
    /// bf16(fp32(base) + fp32(mine rounded) + fp32(peer's rounded value)), each value rounded once.
    bool exchange_add_bf16(int rank, size_t slot, const float* mine, uint16_t* base, size_t n,
                           void* stream, std::string& err);

    /// The split form of `exchange_add_bf16`, for a caller that has work to run between the two halves: `begin`
    /// publishes this card's BF16 rounding of `mine` and releases the lane epochs (no wait), `finish` waits the
    /// peer's epoch, reads both published payloads and combines them into `base`.  The two calls of one round must
    /// be paired on both cards, and every begun round must be finished before `base` is read.  `mine` is not read
    /// by `finish` (the published copy in the slot is), so the caller may reuse that buffer in between.
    bool exchange_add_bf16_begin(int rank, size_t slot, const float* mine, size_t n, void* stream, std::string& err);
    bool exchange_add_bf16_finish(int rank, size_t slot, uint16_t* base, size_t n, void* stream, std::string& err);

    /// FP32 payload exchange: `partial` (device fp32, `n` values) is replaced in place by the sum; with
    /// `bf16_out` the sum is also stored there rounded once (the FP32 payloads' combine is commutative, so both
    /// cards keep equal bits).
    bool exchange_f32(int rank, size_t slot, float* partial, uint16_t* bf16_out, size_t n,
                      void* stream, std::string& err);

    /// True when an exchange gave up waiting for its peer (sticky). Call after the round's stream work.
    bool hang_reported() const;

    /// Marks the pair diverged from the host (an exchange failed to enqueue): sticky, as a kernel's timeout is.
    void poison();

    /// The pair's step gate, currently unwired: no driver posts it, so the exchange's spin budget and the eager
    /// module loading a pair's start sets (CUDA_MODULE_LOADING=EAGER, src/program/cli/startup.cpp) are the
    /// host-stall defenses. The contract, for the day a driver needs it: a card's queued step must not start
    /// spinning for its peer while the peer's host thread is still enqueueing; `post_gate` queues a one-warp
    /// kernel that waits for `gate_release`'s count, and the driver raises the count once both cards' steps are
    /// fully queued. The returned token is the count this gate waits for.
    uint32_t gate_raise();                       ///< the host's plan-to-release: a new step token
    void gate_release();                         ///< the host's release: both cards' work is enqueued
    void post_gate(uint32_t token, void* stream); ///< the queued wait, at a card stream's head

private:
    struct Slot;
    int dev0_ = -1, dev1_ = -1;
    size_t slot_bytes_ = 0;      // the payload region of one slot
    size_t line_bytes_ = 0;      // the release lines of one slot (one 64-byte line per warp lane)
    size_t n_slots_ = 0;
    uint8_t* block_ = nullptr;   // the pinned block (slots' lines, then the hang word)
    uint32_t* hang_ = nullptr;   // the hang word (host view)
    uint32_t* gate_ = nullptr;   // the step gate's release count (host view; the mapped pointer names it on cards too)
    uint32_t* reads_ = nullptr;  // the read-done epochs, in mapped host memory so both cards see them
    uint32_t gate_seq_ = 0;      // the host's issued step tokens
    Slot* slot_views_ = nullptr; // per slot: payload/flag pointers per rank and the per-rank device epochs
    // the device VMM payload: one reserved range per rank, its backing handle, and the range's real extent (the
    // driver's granularity rounds the reservation up, so the mapping is of the rounded length, not `slot_bytes_`)
    uint8_t* payload_va_[2] = {nullptr, nullptr};
    unsigned long long payload_handle_[2] = {0, 0};
    uint64_t payload_span_ = 0;  // the reserved/mapped length of one rank's payload region
    bool dev_payload_ = false;
    // the value count and exchange kind the pair's rank 1 last carried per slot (kind * kKindUnit + count), for
    // the enqueue-time agreement check that the two cards' exchanges say the same geometry (a mismatch wedges
    // the round forever); close() clears it, so a reopen starts the check off
    std::vector<size_t> last_n_;
};

}  // namespace basalt::core
