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

    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
