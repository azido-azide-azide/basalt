// tests/models/qwen4exp/tp_pair_sizing_test.cpp - the Flash-Next TP pair's sizing math, host only: the pair
// hand-off is priced IDENTICALLY by the card and the full geometry (rank 0's publish, rank 1's consume and
// the runtime's one buffer are sized through different geometries - a disagreement would be a silent row
// overrun); the mailbox's two slots of 98304 B carry the pair's widest payloads at T <= kVerifyMaxT (the
// bf16 model-head rows today, the f32 MoE rows the next slice adds, in whole payload-unit multiples); the
// per-window mixer exchange count is even (both cards' slot alternation comes back to slot 0 at every
// window start, so the driver's per-window epi zeroing matches the graphs' baked parities); and the card
// geometry's half widths are what the pair's roles were checked to.
#include "basalt/models/qwen4exp/decode/verify.hpp"             // Verifier::handoff_floats (inline, pure)
#include "basalt/models/qwen4exp/decode/session.hpp"            // session_bytes (pure arithmetic, safe to price)
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"    // basalt::kernels::kVerifyMaxT
#include "basalt/models/qwen4exp/plan/tp_rank.hpp"              // the card plans
#include "basalt/core/tp_mailbox.hpp"                           // kVecBytes/kVecsPerLane (the unit rounding)

#include <cstdio>
#include <string>

using namespace basalt;

namespace {

int failures = 0;
void ok(bool cond, const std::string& what) {
    std::printf("%s %s\n", cond ? "ok  " : "FAIL", what.c_str());
    if (!cond) ++failures;
}

}  // namespace

int main() {
    core::ModelGeometry g;   // the shipped defaults ARE the model's contract (see tp_rank_plan_test)
    std::string err;
    const plan::FnRankPlan p0 = plan::fn_rank_plan(g, 2, 0, err);
    ok(err.empty(), "rank 0 plans clean");
    const plan::FnRankPlan p1 = plan::fn_rank_plan(g, 2, 1, err);
    ok(err.empty(), "rank 1 plans clean");

    // the hand-off is priced identically by either geometry (serve/verify.cpp sizes the runtime buffer with
    // the FULL g; record.hpp's HB computes it per card from the verifier's own geometry)
    ok(core::Verifier::handoff_floats(g) == core::Verifier::handoff_floats(p0.card) &&
       core::Verifier::handoff_floats(p0.card) == core::Verifier::handoff_floats(p1.card),
       "hand-off floats: full == card (the pair's publish and consume agree on the row triple)");

    // the pair's mailbox geometry (serve/verify.cpp bring-up): 2 slots x 98304 B
    constexpr size_t kSlotBytes = 98304;
    const size_t unit = core::TpMailbox::kVecBytes * 32 * (size_t) core::TpMailbox::kVecsPerLane;
    ok(kSlotBytes % unit == 0, "98304 B is a whole number of mailbox payload units (no rounding grows it)");
    const int64_t T = basalt::kernels::kVerifyMaxT;
    const int64_t bf16_rows = T * g.n_head * g.head_dim * 2;      // 8 * 6144 * 2: the mixer exchange (32c-2)
    const int64_t f32_moe = T * g.n_embd * 4;                     // 8 * 2560 * 4: the MoE partial (32c-4)
    ok(bf16_rows <= (int64_t) kSlotBytes, "a slot carries the widest bf16 model-head rows at T<=max");
    ok(f32_moe <= (int64_t) kSlotBytes, "the same slot carries the f32 MoE partial rows at T<=max");
    const int64_t hand = basalt::kernels::kVerifyMaxT * core::Verifier::handoff_floats(g) * 4;
    ok(hand > 0, "the pair hand buffer prices to " + std::to_string(hand) + " B (mapped, one block, both windows)");

    // one mixer exchange per layer (the pair's assembly before each replicated output projection):
    // every window queues n_layers exchanges, an EVEN count -> the 2-slot alternation returns to slot 0 at
    // the next window on both cards, which is what the driver's per-window epi reset (forward_window law)
    // assumes.  The plan's card halves never change this count (the mixers are one per layer per card).
    const int64_t exch = g.n_layers;
    ok(exch % 2 == 0, "per-window mixer exchanges are even (slot parity re-aligns at every window start)");
    ok(p0.card.n_head * 2 == g.n_head && p1.card.ssm_value_dim * 2 == g.ssm_value_dim,
       "card halves are exact (the roles the pair was loaded and checked against)");

    // rank 1's session is priced on the CARD geometry (serve/verify.cpp's init_pair_rank1 guard line):
    // session_bytes is pure arithmetic, safe to price on the host - and the card's half-width KV must
    // show up as a strictly smaller carve than the full geometry's.
    const uint64_t full_s = core::session_bytes(g, 65536, 10, 0, -1);
    const uint64_t card_s = core::session_bytes(p0.card, 65536, 10, 0, -1);
    std::printf("     (session price at 65536 cells: full %llu MiB, card %llu MiB)\n",
                (unsigned long long) (full_s >> 20), (unsigned long long) (card_s >> 20));
    ok(card_s < full_s && card_s > 0, "a card-geometry session is strictly cheaper (rank 1's carve fits the pair's card)");

    std::printf("%s: tp pair sizing %s\n", failures ? "FAILURES" : "all ok", failures ? "FAILED" : "passed");
    return failures != 0;
}
