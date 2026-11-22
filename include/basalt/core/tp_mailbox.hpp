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
// The payloads cross PCIe as card-to-host reads of pinned memory: a card never needs to name the other card's
// memory, so this works whether or not the driver granted peer access.
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
    static constexpr int kVecsPerLane = 2;

    TpMailbox() = default;
    ~TpMailbox();
    TpMailbox(const TpMailbox&) = delete;
    TpMailbox& operator=(const TpMailbox&) = delete;

    /// Opens `slots` slots, each holding `max_bytes`, for the ordered card pair (`rank0_dev`, `rank1_dev`).
    /// Page-locked memory is allocated for every slot and mapped into both cards' address spaces; per-warp
    /// release lines live at the front of each slot and a shared hang word at the front of the block.
    /// `max_bytes` is rounded up to a multiple of `kVecBytes * 32 * kVecsPerLane` (the whole-geometry unit).
    bool open(int rank0_dev, int rank1_dev, size_t slots, size_t max_bytes, std::string& err);
    void close();

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
    uint8_t* block_ = nullptr;   // the whole pinned block (slots' lines, then slots' payloads, then the hang word)
    uint32_t* hang_ = nullptr;   // the hang word (host view)
    uint32_t* gate_ = nullptr;   // the step gate's release count (host view; the mapped pointer names it on cards too)
    uint32_t gate_seq_ = 0;      // the host's issued step tokens
    Slot* slot_views_ = nullptr; // per slot: payload/flag pointers per rank and the per-rank device epochs
    // the value count and exchange kind the pair's rank 1 last carried per slot (kind * kKindUnit + count), for
    // the enqueue-time agreement check that the two cards' exchanges say the same geometry (a mismatch wedges
    // the round forever); close() clears it, so a reopen starts the check off
    std::vector<size_t> last_n_;
};

}  // namespace basalt::core
