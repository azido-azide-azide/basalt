// src/models/qwen4exp/plan/tp_rank.cpp - see the header for what the cut is; this is the arithmetic.
#include "basalt/models/qwen4exp/plan/tp_rank.hpp"

namespace basalt::plan {

std::string fn_tp_check(const core::ModelGeometry& g, int tp) {
    if (tp != 2) return "qwen4exp: tensor parallelism supports 2 cards for now";
    if (g.n_head % tp) return "qwen4exp: query heads must divide the cards";
    if (g.n_head_kv % tp) return "qwen4exp: attention key-value heads must divide the cards";
    if ((g.n_head / tp) % (g.n_head_kv / tp)) return "qwen4exp: the per-card head grouping would not be whole";
    if (g.ssm_k_heads % tp) return "qwen4exp: GDN key heads must divide the cards";
    if (g.ssm_v_heads % tp) return "qwen4exp: GDN value heads must divide the cards";
    if ((g.ssm_v_heads / tp) % (g.ssm_k_heads / tp)) return "qwen4exp: the per-card GDN value grouping would not be whole";
    // row slices land at head boundaries: multiples of head_dim (256) and state (128) times head counts.
    if ((g.n_head / tp) * g.head_dim % 256) return "qwen4exp: the per-card attention rows are not whole 256-value blocks";
    if ((g.ssm_v_heads / tp) * g.ssm_state_size % 128) return "qwen4exp: the per-card GDN rows are not whole 128-value blocks";
    if (g.qsa_interval <= 0 || g.n_layers % g.qsa_interval) return "qwen4exp: the layer mix does not split into whole QSA intervals";
    return {};
}

FnRankPlan fn_rank_plan(const core::ModelGeometry& g, int tp, int rank, std::string& err) {
    FnRankPlan p;
    if (err = fn_tp_check(g, tp); !err.empty()) return p;
    if (rank < 0 || rank >= tp) {
        err = "qwen4exp: card rank " + std::to_string(rank) + " is not in a " + std::to_string(tp) + "-card group";
        return p;
    }
    p.tp = tp;
    p.rank = rank;
    p.full = g;
    p.card = g;
    p.card.n_head = g.n_head / tp;
    p.card.n_head_kv = g.n_head_kv / tp;
    p.card.ssm_k_heads = g.ssm_k_heads / tp;
    p.card.ssm_v_heads = g.ssm_v_heads / tp;
    p.card.ssm_conv_channels = g.ssm_conv_channels / tp;   ///< (2·nk + nv)·s with the card's heads
    p.card.ssm_value_dim = g.ssm_value_dim / tp;
    p.gdn = core::GdnHeadSlice{static_cast<int>(g.ssm_k_heads), static_cast<int>(g.ssm_v_heads),
                               static_cast<int>(g.ssm_state_size), static_cast<int>(rank * g.ssm_k_heads / tp),
                               static_cast<int>(g.ssm_k_heads / tp)};
    if (err = p.check(); !err.empty()) return {};
    return p;
}

std::string FnRankPlan::check() const {
    if (gdn.c() != card.ssm_conv_channels) return "fn plan: the card's GDN channel count disagrees";
    if (gdn.zv() != card.ssm_value_dim) return "fn plan: the card's GDN value width disagrees";
    if (static_cast<int64_t>(gdn.nv()) != card.ssm_v_heads) return "fn plan: the card's GDN value heads disagree";
    if (aq() % 256 || kv() % 256) return "fn plan: the attention head runs are not whole 256-value blocks";
    return {};
}

std::vector<core::GdnRun> FnRankPlan::qkv_rows() const { return core::gdn_qkv_runs(gdn); }
std::vector<core::GdnRun> FnRankPlan::v_rows(int64_t per) const { return core::gdn_vhead_runs(gdn, per); }
std::vector<core::GdnRun> FnRankPlan::q_rows() const {
    return {{static_cast<int64_t>(rank) * aq() * 2, aq() * 2}};   // [query | gate] pairs, head order
}
std::vector<core::GdnRun> FnRankPlan::kv_rows() const { return {{static_cast<int64_t>(rank) * kv(), kv()}}; }

FnPlace FnRankPlan::placement(const std::string& name, int64_t layer, bool* unknown) const {
    const bool qsa = is_qsa_layer(full, layer);
    if (unknown) *unknown = false;
    if (name == "attn_output.weight" || name == "ssm_out.weight") return FnPlace::Replicated;   // partials exchange
    if (name == "attn_q.weight" || name == "attn_k.weight" || name == "attn_v.weight")
        return qsa ? FnPlace::RankRows : FnPlace::Replicated;
    if (name == "attn_q_norm.weight" || name == "attn_k_norm.weight") return FnPlace::Replicated;
    if (name == "attn_qkv.weight" || name == "attn_gate.weight" || name == "ssm_conv1d.weight")
        return FnPlace::RankRows;
    if (name == "ssm_alpha.weight" || name == "ssm_beta.weight" || name == "ssm_dt.bias" || name == "ssm_a" ||
        name == "ssm_norm.weight")
        return FnPlace::Replicated;   // 1-D vectors of 48 values / state width: kernels take the head range
    if (name.rfind("hc_", 0) == 0 || name.rfind("indexer.", 0) == 0) return FnPlace::Replicated;
    if (name == "ffn_gate_inp.weight" || name == "ffn_gate_inp_shexp.weight") return FnPlace::Replicated;
    if (name == "ffn_gate_shexp.weight" || name == "ffn_up_shexp.weight" || name == "ffn_down_shexp.weight")
        return FnPlace::SharedOwner;
    if (unknown) *unknown = true;
    return FnPlace::Replicated;
}

WeightTable::RowSlices fn_row_slices(const FnRankPlan& p) {
    WeightTable::RowSlices m;
    const auto qkv = p.qkv_rows();
    const auto gate = p.v_rows(p.full.ssm_state_size);
    const auto q = p.q_rows();
    const auto kv = p.kv_rows();
    auto add = [&m](const std::string& name, const std::vector<core::GdnRun>& runs) {
        std::vector<std::pair<int64_t, int64_t>> v;
        for (const auto& r : runs) v.emplace_back(r.start, r.len);
        m[name] = std::move(v);
    };
    for (int64_t l = 0; l < p.full.n_layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        if (is_qsa_layer(p.full, l)) {
            add(b + "attn_q.weight", q);
            add(b + "attn_k.weight", kv);
            add(b + "attn_v.weight", kv);
        } else {
            add(b + "attn_qkv.weight", qkv);
            add(b + "attn_gate.weight", gate);
            add(b + "ssm_conv1d.weight", qkv);
        }
    }
    return m;
}

bool FnRankPlan::wants_columns(const std::string& name) {
    // policy: this plan slices NOTHING by columns; anything named like a column cut is a caller bug.
    return name == "attn_output.weight" || name == "ssm_out.weight";
}

}  // namespace basalt::plan
