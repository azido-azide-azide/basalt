// tests/models/qwen4exp/tp_rank_plan_test.cpp - the Flash-Next TP card plan, host only: the checks accept the
// real geometry and refuse the broken ones, the two cards' ROW runs partition every sliced axis exactly once
// at whole-block head boundaries, no tensor role asks for a column slice (policy, per the arena layout),
// expert parity and shared ownership cover every blob exactly once, and every manifest tensor role has a
// placement (with the unknowns named).
#include "basalt/models/qwen4exp/plan/tp_rank.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace basalt;
using namespace basalt::plan;

namespace {

int failures = 0;
void ok(bool cond, const std::string& what) {
    std::printf("%s %s\n", cond ? "ok  " : "FAIL", what.c_str());
    if (!cond) ++failures;
}

void cover(const std::vector<core::GdnRun>& runs, int64_t axis, const std::string& what) {
    std::vector<char> seen(static_cast<size_t>(axis), 0);
    bool good = true;
    for (const auto& r : runs)
        for (int64_t i = r.start; i < r.start + r.len; ++i) {
            if (i < 0 || i >= axis || seen[static_cast<size_t>(i)]) { good = false; break; }
            seen[static_cast<size_t>(i)] = 1;
        }
    int64_t total = 0;
    for (const auto& r : runs) total += r.len;
    ok(good && total == axis, what);
}

}  // namespace

int main() {
    core::ModelGeometry g;   // the shipped defaults ARE the model's contract
    ok(fn_tp_check(g, 2).empty(), "the real geometry passes the TP check");
    ok(!fn_tp_check(g, 3).empty(), "3 cards refused (2 only for now)");
    {
        core::ModelGeometry bad = g;
        bad.n_head_kv = 3;
        ok(!fn_tp_check(bad, 2).empty(), "odd kv heads refused");
        core::ModelGeometry bad2 = g;
        bad2.ssm_v_heads = 46;
        ok(!fn_tp_check(bad2, 2).empty(), "GDN value grouping not whole refused");
    }

    std::string err;
    FnRankPlan p0 = fn_rank_plan(g, 2, 0, err);
    ok(err.empty(), "rank 0 plans clean");
    FnRankPlan p1 = fn_rank_plan(g, 2, 1, err);
    ok(err.empty(), "rank 1 plans clean");
    ok(p0.card.n_head == 12 && p0.card.n_head_kv == 1, "half-wide QSA mixers per card");
    ok(p0.card.ssm_k_heads == 8 && p0.card.ssm_v_heads == 24, "GDN 8 keys / 24 values per card");
    ok(p0.check().empty(), "rank 0 self-check clean");
    ok(p1.check().empty(), "rank 1 self-check clean");

    // axis partitions across the two cards, every sliced axis exactly once
    {
        std::vector<core::GdnRun> both = p0.q_rows();
        auto r1 = p1.q_rows();
        both.insert(both.end(), r1.begin(), r1.end());
        cover(both, 2 * g.n_head * g.head_dim, "attn_q rows: both cards cover the axis exactly");
        both = p0.kv_rows();
        auto kr = p1.kv_rows();
        both.insert(both.end(), kr.begin(), kr.end());
        cover(both, g.n_head_kv * g.head_dim, "attn_k/v rows cover exactly");
        both = p0.qkv_rows();
        auto qr = p1.qkv_rows();
        both.insert(both.end(), qr.begin(), qr.end());
        cover(both, 2 * g.ssm_k_heads * g.ssm_state_size + g.ssm_value_dim, "GDN qkv rows cover exactly");
        both = p0.v_rows(g.ssm_state_size);
        auto vr = p1.v_rows(g.ssm_state_size);
        both.insert(both.end(), vr.begin(), vr.end());
        cover(both, g.ssm_value_dim, "GDN gate/z/state rows cover exactly");
    }
    for (const auto& c : p0.q_rows()) ok(c.start % 256 == 0 && c.len % 256 == 0, "attention row runs 256-aligned");
    for (const auto& c : p0.v_rows(g.ssm_state_size))
        ok(c.start % 128 == 0 && c.len % 128 == 0, "GDN value row runs 128-aligned");

    // the loader map: every head-sliced role of every layer, sized to the shapes the layout contract pins
    {
        core::WeightTable::RowSlices m0 = fn_row_slices(p0), m1 = fn_row_slices(p1);
        ok(m0.size() == 3 * g.n_layers && m1.size() == m0.size(), "the slice map covers 3 roles x 48 layers");
        int64_t rows0 = 0, rows1 = 0;
        for (const auto& kv : m0)
            for (const auto& r : kv.second) rows0 += r.second;
        for (const auto& kv : m1)
            for (const auto& r : kv.second) rows1 += r.second;
        const int64_t want = 12 * (2 * g.n_head * g.head_dim + 2 * g.n_head_kv * g.head_dim) +
                             36 * (2 * g.ssm_conv_channels + g.ssm_value_dim);
        ok(rows0 == want / 2 && rows1 == want - want / 2, "both cards' slices together cover exactly the sliced rows");
        ok(m0.count("blk.3.attn_q.weight") && m0.count("blk.0.attn_qkv.weight") && m0.count("blk.0.ssm_conv1d.weight"),
           "the map names per-kind roles on the right layers");
        ok(!m0.count("blk.3.ssm_out.weight") && !m0.count("blk.0.hc_ffn_up.weight"),
           "replicated roles appear in no slice");
    }

    // the plan never asks for a column slice
    ok(!FnRankPlan::wants_columns("attn_qkv.weight") && !FnRankPlan::wants_columns("ssm_alpha.weight"),
       "row-sliced tensors want no columns");

    // experts and shared ownership: every blob owned exactly once
    int64_t owned0 = 0, owned1 = 0;
    for (int64_t e = 0; e < g.n_expert; ++e) {
        const int n = (p0.owns_expert(e) ? 1 : 0) + (p1.owns_expert(e) ? 1 : 0);
        owned0 += p0.owns_expert(e);
        owned1 += p1.owns_expert(e);
        if (n != 1) { ok(false, "an expert blob has two owners or none"); break; }
    }
    ok(owned0 == 256 && owned1 == 256, "expert parity splits 512 into 256+256");
    int shared0 = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) shared0 += p0.owns_shared(l) ? 1 : 0;
    ok(shared0 == g.n_layers / 2, "shared experts alternate owners");

    // the manifest roles all have placements, and the output projections are REPLICATED (partials exchange)
    const std::vector<std::string> qsa = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                          "attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_proj.weight",
                                          "indexer.k_proj.weight", "hc_attn_up.weight", "hc_ffn_down.weight",
                                          "ffn_gate_inp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                          "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight"};
    const std::vector<std::string> gdn = {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ssm_alpha.weight",
                                          "ssm_beta.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a",
                                          "ssm_norm.weight"};
    for (const auto& n : qsa) {
        bool unk = false;
        p0.placement(n, 3, &unk);
        ok(!unk, "QSA role placed: " + n);
    }
    for (const auto& n : gdn) {
        bool unk = false;
        p0.placement(n, 0, &unk);
        ok(!unk, "GDN role placed: " + n);
    }
    ok(p0.placement("attn_output.weight", 3) == FnPlace::Replicated, "attn_output replicated (partials over the mailbox)");
    ok(p0.placement("ssm_out.weight", 0) == FnPlace::Replicated, "ssm_out replicated (partials over the mailbox)");
    ok(p0.placement("ffn_down_shexp.weight", 1) == FnPlace::SharedOwner, "shared expert owned by layer parity");
    ok(p0.placement("hc_ffn_up.weight", 0) == FnPlace::Replicated, "the hc stacks stay replicated");
    ok(p0.placement("indexer.q_proj.weight", 3) == FnPlace::Replicated, "the QSA indexer stays replicated");
    ok(p0.placement("attn_qkv.weight", 0) == FnPlace::RankRows && p0.placement("attn_q.weight", 3) == FnPlace::RankRows,
       "input-side projections slice rows by heads");
    bool unk = false;
    p0.placement("mystery.thing.weight", 0, &unk);
    ok(unk, "unknown names are named as unknown");

    // ---- the head map a PAIR RUNTIME owes the plan (tp-10) ---------------------------------------------
    // The plan cuts GDN value heads by KEY-head group (core::gdn_head_split: value head v reads key head
    // v % h_k), so a card's packed local head p is model head (p / nk) * h_k + k0 + p % nk - NOT
    // rank * nv + p.  Everywhere the pair touches a REPLICATED full-width object (a per-head parameter row
    // of ssm_alpha / ssm_beta / ssm_dt.bias / ssm_a, or the replicated ssm_out's columns) it must address
    // that model head.  Two places did the contiguous arithmetic instead, which is the pair's token salad:
    // the recurrence's `hv0 = rank * HV` offset, and tp_assemble's `off = rank * half`.
    {
        const core::GdnHeadSlice* cards[2] = {&p0.gdn, &p1.gdn};
        ok(p0.gdn.h_k == 16 && p0.gdn.h_v == 48 && p0.gdn.nk == 8 && p0.gdn.nv() == 24 && p0.gdn.s == 128,
           "the pair's GDN slice: 16 model key heads with 8 per card, 48 value heads with 24 per card");
        std::vector<char> named((size_t) g.ssm_v_heads, 0);
        bool once = true;
        for (const core::GdnHeadSlice* c : cards)
            for (int64_t p = 0; p < c->nv(); ++p) {
                const int64_t h = core::gdn_local_head(*c, p);
                if (h < 0 || h >= (int64_t) g.ssm_v_heads || named[(size_t) h]++) once = false;
            }
        ok(once, "the two cards' packed local heads name every model value head exactly once");

        bool agrees = true;
        for (const core::GdnHeadSlice* c : cards) {
            int64_t at = 0;
            for (const core::GdnRun& r : core::gdn_vhead_runs(*c, 1))
                for (int64_t i = 0; i < r.len; ++i, ++at)
                    if (core::gdn_local_head(*c, at) != r.start + i) agrees = false;
        }
        ok(agrees, "gdn_local_head reproduces gdn_vhead_runs' packed order, element for element");

        int moved = 0;
        for (int r = 0; r < 2; ++r)
            for (int64_t p = 0; p < cards[r]->nv(); ++p)
                if (core::gdn_local_head(*cards[r], p) != (int64_t) r * cards[r]->nv() + p) ++moved;
        ok(moved == 32, "a key-head-group cut moves 32 of the 48 heads off the contiguous block (the damage "
                        "a rank * nv offset did)");

        bool placed = true;
        std::vector<char> owned((size_t) g.ssm_value_dim, 0);
        for (int r = 0; r < 2; ++r) {
            const core::GdnHeadSlice& c = *cards[r];
            int64_t sg = 0, dg = 0, base = 0;
            core::gdn_place_groups(c, sg, dg, base);
            ok(sg == (int64_t) c.nk * c.s && dg == (int64_t) c.h_k * c.s && base == (int64_t) c.k0 * c.s,
               "the group strides are nk * s, h_k * s and k0 * s");
            for (int64_t p = 0; p < c.nv() && placed; ++p)
                for (int64_t e = 0; e < c.s; ++e)
                    if (core::gdn_grouped_src_col(core::gdn_local_col(c, p) + e, sg, dg, base) != p * c.s + e)
                        placed = false;
            for (int64_t col = 0; col < (int64_t) c.h_v * c.s; ++col) {
                const int64_t src = core::gdn_grouped_src_col(col, sg, dg, base);
                if (src < 0) continue;
                if (src >= (int64_t) c.nv() * c.s || owned[(size_t) col]++) placed = false;
            }
        }
        ok(placed, "every card head lands on its OWN columns of the full row, and no column lands twice");
        bool covers = owned.size() == (size_t) g.ssm_value_dim;
        for (char o : owned)
            if (o != 1) covers = false;
        ok(covers, "the pair's grouped placements partition the replicated projection's column axis exactly once");

        {   // with one value head per key head the cut IS contiguous: the two forms must agree, so this fix
            // cannot change a model whose GDN has no key-head grouping
            const core::GdnHeadSlice one{8, 8, 128, 4, 4};
            int64_t sg = 0, dg = 0, base = 0;
            core::gdn_place_groups(one, sg, dg, base);
            bool same = true;
            for (int64_t col = 0; col < (int64_t) one.h_v * one.s; ++col) {
                const int64_t grouped = core::gdn_grouped_src_col(col, sg, dg, base);
                const int64_t plain = (col >= base && col < base + (int64_t) one.nv() * one.s) ? col - base : -1;
                if (grouped != plain) same = false;
            }
            ok(same, "with h_v == h_k the grouped placement and rank * half are the same function");
        }
    }

    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
