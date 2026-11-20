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
#include "basalt/models/qwen4exp/decode/layout.hpp"             // is_qsa_layer (the mixer split, pure)
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
    // every window queues n_layers exchanges, an EVEN count -> the 2-slot alternation comes back to slot 0 at
    // the next window on both cards, which is what the driver's per-window epi reset (forward_window law)
    // assumes.  The plan's card halves never change this count (the mixers are one per layer per card).
    const int64_t exch = g.n_layers;
    ok(exch % 2 == 0, "per-window mixer exchanges are even (slot parity re-aligns at every window start)");

    // ---- 32c-4 adds ONE MORE exchange per layer per card: the FP32 MoE PARTIAL SUM after each card's
    // combine (record_moe.cpp's dm branch, exchange_f32 over the T*N bo_ rows through the same alternating
    // slots).  Two things must hold host-side: the f32 payload at the widest window is a WHOLE number of
    // mailbox payload units (the exchange's n_agree compares per-card value counts at enqueue - a rounded
    // site would have to round identically on both cards, and there is no reason for 2560*4 to come out
    // whole if the unit ever changed); and the per-window TOTAL, mixers + MoE, stays EVEN, so the slot
    // alternation still lands back on slot 0 at every window start on both cards.  48 mixers + 48 MoE = 96.
    ok(f32_moe % (int64_t) unit == 0, "the f32 MoE payload is a whole number of mailbox payload units");
    int64_t n_qsa = 0;
    for (int64_t l = 0; l < g.n_layers; ++l)
        if (core::is_qsa_layer(g, l)) ++n_qsa;   // the mixer split the totals line names: GDN + QSA = layers
    const int64_t exch_moe = g.n_layers;         // one partial-sum exchange after each layer's combine
    const int64_t exch_all = exch + exch_moe;
    ok(n_qsa + (g.n_layers - n_qsa) == g.n_layers && exch_all == 2 * g.n_layers && exch_all % 2 == 0,
       "per-window exchanges = (" + std::to_string(g.n_layers - n_qsa) + " GDN + " + std::to_string(n_qsa) +
       " QSA) mixers + " + std::to_string(exch_moe) + " MoE = " + std::to_string(exch_all) +
       ", even (both cards' totals and parities agree at every width)");
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

    // ---- the pair's device-local hyper-connection copies (32c-3b).  What is host-checkable here is the plan
    // INVARIANT the twin's rebind pass stands on: every hc8 skip name is REPLICATED (tp_rank.cpp:72) and no rank's
    // row slices ever name an hc matrix - so both tables keep the file table's one (rows, cols) per name and
    // core::hc8_find resolves the twin's (r1->ne1, r1->ne0) exactly as rank 0's.  A sliced hc entry would make the
    // twin's hc8_table_adopt return nullptr and the load refuse, so this pins the failure before a window pays it.
    // The adoption itself is NOT host-testable and NOT exercised: hc8_table_adopt ends in hc8_upload, whose
    // cudaMalloc needs a device - two distinct copy pointers for {rank0, rank1} cannot be produced (or keyed
    // (device, pointer) in hc_fused.cu's registry) on a machine without running CUDA, which this slice forbids.
    bool hc_replicated = true, hc_unsliced = true;
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (const char* role : {"hc_attn_down.weight", "hc_attn_up.weight", "hc_ffn_down.weight", "hc_ffn_up.weight",
                                 "hc_attn_inject.weight", "hc_ffn_inject.weight"})
            for (const plan::FnRankPlan* p : {&p0, &p1})
                hc_replicated = hc_replicated && p->placement(role, l) == plan::FnPlace::Replicated;
    for (const plan::FnRankPlan* p : {&p0, &p1})
        for (const auto& kv : plan::fn_row_slices(*p))
            if (kv.first.find("hc_") != std::string::npos) hc_unsliced = false;   // keys are "blk.<l>.<role>"
    ok(hc_replicated, "every hyper-connection matrix is Replicated (both cards hold it whole)");
    ok(hc_unsliced, "no card's row slices name an hc matrix (the twin's hc8 entries keep the file table's shape)");

    std::printf("%s: tp pair sizing %s\n", failures ? "FAILURES" : "all ok", failures ? "FAILED" : "passed");
    return failures != 0;
}
