// src/models/qwen35/shard.hpp - how a qwen35 pack's matrices split across a tensor-parallel pair: the heads one card
// owns, per tensor role, and the runs that lay the card's copy out.  Free of the runtime and of CUDA, so a CPU test
// can check the whole plan against a pack's own tensor table.
//
// A card owns whole heads, which is what makes the cut legal for both mixers.  Attention: its query heads (each with
// its output gate stacked behind them) and their key/value heads in even groups, so a card's queries meet their own
// keys.  The GDN by key-head group (core/gdn_head_split.hpp): value head v reads key head v % h_k, so a card owns
// whole key-head groups and its slice is a smaller GDN layer of its own - qkv rows and convolution channels, the
// gate's rows, the output's columns, the alpha/beta rows, dt and a, all in the packed order that header gives.
// The FFN shares its intermediate width.  What multiplies a residual (the norms) and the state's group norm are the
// same on both cards; the drafter's bytes and the token embedding stay with the card that reads them.
#pragma once

#include "basalt/core/gdn_head_split.hpp"
#include "basalt/dense/model.hpp"
#include "basalt/dense/weight_shard.hpp"

#include <string>
#include <utility>
#include <vector>

namespace basalt::models::qwen35 {

/// The tensors one card holds alone rather than a shard of (the drafter; the embedding is read from page-locked RAM).
inline bool tp_one_card(const std::string& name) {
    return name == "token_embd.weight" || name.starts_with("dflash.") || name.starts_with("mtp.");
}

/// False with `err` when the model's heads and widths do not share out evenly across `width` cards.
inline bool tp_geometry(const dense::Geometry& g, int width, std::string& err) {
    const auto bad = [&](bool off, const char* what) {
        if (!off) return false;
        err = "qwen35: the model's " + std::string(what) + " do not divide between " + std::to_string(width) + " cards";
        return true;
    };
    if (bad(width < 2, "the tensor-parallel width")) return false;
    if (bad(g.n_head % width != 0 || g.n_kv % width != 0, "attention heads")) return false;
    if (bad(g.gdn_k_heads % width != 0 || g.gdn_v_heads % g.gdn_k_heads != 0, "delta-net heads")) return false;
    if (bad(g.n_ff % width != 0, "feed-forward width")) return false;
    if (bad(g.n_vocab % width != 0, "vocabulary")) return false;
    return true;
}

/// C++20 has no std::string::ends_with.
inline bool tp_ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// `name` without its `blk.<n>.`: the role a layer tensor plays.
inline std::string tp_role(const std::string& name) {
    if (!name.starts_with("blk.")) return name;
    const size_t dot = name.find('.', 4);
    return dot == std::string::npos ? name : name.substr(dot + 1);
}

/// the key heads one card of `width` owns, and the value heads that read them.
inline core::GdnHeadSlice tp_gdn_slice(const dense::Geometry& g, int width, int rank) {
    const int nk = g.gdn_k_heads / width;
    return {g.gdn_k_heads, g.gdn_v_heads, g.gdn_state, rank * nk, nk};
}

inline std::vector<dense::ShardRun> tp_runs(const std::vector<core::GdnRun>& runs) {
    std::vector<dense::ShardRun> out;
    out.reserve(runs.size());
    for (const core::GdnRun& r : runs) out.push_back({r.start, r.len});
    return out;
}

/// The card `rank` of `width` holds of the pack tensor `name`, whose whole matrix is `rows` x `cols` as the loader
/// reads them.  `shard.dim` stays `None` for the tensors every card holds whole.  A layer or head tensor the rules
/// here do not name is an error: a pack with a matrix this code does not know gets refused, not run with the whole
/// of it on both cards.
inline bool tp_shard(const dense::Geometry& g, const std::string& name, int64_t rows, int64_t cols, int width, int rank,
                     dense::MatrixShard& shard, std::string& err) {
    shard = dense::MatrixShard{};
    shard.rows = rows;
    shard.cols = cols;
    if (tp_one_card(name)) return true;
    const std::string role = tp_role(name);
    if (role == "attn_norm.weight" || role == "post_attention_norm.weight" || role == "attn_q_norm.weight" ||
        role == "attn_k_norm.weight" || role == "ssm_norm.weight" || role == "output_norm.weight")
        return true;   // it multiplies a whole residual or one head's width: both cards hold it whole
    const auto wrong = [&](int64_t want_rows, int64_t want_cols) {
        err = "qwen35: " + name + " is " + std::to_string((long long) rows) + " x " + std::to_string((long long) cols) +
              "; tensor parallel needs " + std::to_string((long long) want_rows) + " x " + std::to_string((long long) want_cols);
        return false;
    };
    const int64_t H = g.n_embd, AQ = (int64_t) g.n_head * g.head_dim, KV = (int64_t) g.n_kv * g.head_dim;
    const int64_t VD = g.value_width(), C = g.conv_channels();
    const core::GdnHeadSlice sl = tp_gdn_slice(g, width, rank);
    const auto qkv = [&] { return tp_runs(core::gdn_qkv_runs(sl, g.gdn_state)); };
    const auto vheads = [&] { return tp_runs(core::gdn_vhead_runs(sl, g.gdn_state)); };
    const auto gates = [&] { return tp_runs(core::gdn_vhead_runs(sl, 1)); };
    const auto rows_of = [&](std::vector<dense::ShardRun> runs) {
        shard.dim = dense::ShardDim::Rows;
        shard.runs = std::move(runs);
        return true;
    };
    const auto cols_of = [&](std::vector<dense::ShardRun> runs) {
        shard.dim = dense::ShardDim::Cols;
        shard.runs = std::move(runs);
        return true;
    };
    if (role == "attn_qkv.weight") return rows == C && cols == H ? rows_of(qkv()) : wrong(C, H);
    if (role == "attn_gate.weight") return rows == VD && cols == H ? rows_of(vheads()) : wrong(VD, H);
    if (role == "ssm_out.weight") return rows == H && cols == VD ? cols_of(vheads()) : wrong(H, VD);
    if (role == "ssm_conv1d.weight") return rows == C && cols == g.gdn_conv ? rows_of(qkv()) : wrong(C, g.gdn_conv);
    if (role == "ssm_alpha.weight" || role == "ssm_beta.weight")
        return rows == g.gdn_v_heads && cols == H ? rows_of(gates()) : wrong(g.gdn_v_heads, H);
    if (role == "ssm_a" || role == "ssm_dt.bias")
        return rows == g.gdn_v_heads && cols == 1 ? rows_of(gates()) : wrong(g.gdn_v_heads, 1);
    // the query projection stacks its gates behind all its queries, so a card's rows are its heads' queries and, at the
    // same offset behind them, those heads' gates - the same shape of matrix as the whole one
    if (role == "attn_q.weight") {
        if (rows != 2 * AQ || cols != H) return wrong(2 * AQ, H);
        std::vector<dense::ShardRun> q = dense::unit_runs(g.n_head, g.head_dim, width, rank);
        dense::append_runs(q, dense::unit_runs(g.n_head, g.head_dim, width, rank), AQ);
        return rows_of(std::move(q));
    }
    if (role == "attn_k.weight" || role == "attn_v.weight")
        return rows == KV && cols == H ? rows_of(dense::unit_runs(g.n_kv, g.head_dim, width, rank)) : wrong(KV, H);
    if (role == "attn_output.weight")
        return rows == H && cols == AQ ? cols_of(dense::unit_runs(g.n_head, g.head_dim, width, rank)) : wrong(H, AQ);
    if (role == "ffn_gate.weight" || role == "ffn_up.weight")
        return rows == g.n_ff && cols == H ? rows_of(dense::unit_runs(g.n_ff, 1, width, rank)) : wrong(g.n_ff, H);
    if (role == "ffn_down.weight")
        return rows == H && cols == g.n_ff ? cols_of(dense::unit_runs(g.n_ff, 1, width, rank)) : wrong(H, g.n_ff);
    // A matrix's FP32 tensor scale is one value every card applies to its own shard; only the head's scales are per row.
    if (role != "output.scale" && tp_ends_with(role, ".scale")) return true;
    if (role == "output.weight") return rows == g.n_vocab && cols == H ? rows_of(dense::unit_runs(g.n_vocab, 1, width, rank)) : wrong(g.n_vocab, H);
    if (role == "output.scale") return rows == g.n_vocab && cols == 1 ? rows_of(dense::unit_runs(g.n_vocab, 1, width, rank)) : wrong(g.n_vocab, 1);
    err = "qwen35: tensor parallel has no rule for " + name;
    return false;
}

}  // namespace basalt::models::qwen35
