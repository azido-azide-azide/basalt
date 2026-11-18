#pragma once
// include/basalt/models/qwen4exp/plan/tp_rank.hpp - the tensor-parallel card plan for the Flash-Next MoE model
// (qwen4exp): what one card of a `tp`-card group holds, by tensor role.  The cut follows the proven reference
// design for this model family: the mixer OUTPUT projections (QSA attn_output, GDN ssm_out) are REPLICATED —
// both cards hold the whole weight and each multiplies its own head columns' activations, producing a PARTIAL
// sum that the block's one BF16 mailbox exchange turns into the same bits on both cards.  Replicating the
// weight (instead of slicing its columns) is what the arena's layout forces — quantized tensors sit as ONE
// interleaved block stream (tests/core/tp_col_probe.cpp measures the refusal) — and it costs only VRAM.  The
// hc streams and the router are replicated too.  The card's GDN slice is its key-head range
// (core::GdnHeadSlice): its activations carry the card's value heads.  The shared expert stays whole and
// alternates owner by layer parity; routed experts are partitioned WHOLE by expert-id parity so each card
// streams only its half over its own link, and the MoE block output is one FP32 exchange.  Head, embedding,
// PLE and MTP belong to rank 0 alone.  Every weight slice this plan asks the loader to build is a ROW range
// at a head boundary (32/128/256-aligned — whole quant blocks in every arena form).  No kernel bytes move
// here: this is the plan; the loader and the pair driver act on it.
#include "basalt/core/gdn_head_split.hpp"
#include "basalt/core/weights.hpp"   // WeightTable::RowSlices
#include "basalt/models/qwen4exp/decode/layout.hpp"   // core::ModelGeometry

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::plan {

/// Where a tensor (or expert stream) sits in a TP group.
enum class FnPlace : uint8_t {
    Replicated,    ///< both cards hold it whole and run it on their identical inputs
    RankRows,      ///< row-parallel: this card's head runs of the output axis (a row-range arena slice)
    SharedOwner,   ///< the layer's shared expert, whole, on owner = layer % tp
    Rank0Only,     ///< head, embedding, PLE, MTP: rank 0
};

/// One card's plan.  `card` is the geometry the card's kernels see: half the query/key/value heads and the
/// card's GDN key-head range; everything else is the full model's.
struct FnRankPlan {
    int tp = 2, rank = 0;
    core::ModelGeometry full;
    core::ModelGeometry card;
    core::GdnHeadSlice gdn;   ///< the card's GDN key-head range of the full layer

    std::string check() const;   // empty when the plan is consistent

    // ---- run providers over the ROW axis (core::GdnRun; the only axis this plan slices) ----
    std::vector<core::GdnRun> qkv_rows() const;   ///< GDN fused qkv rows and the conv's channels
    std::vector<core::GdnRun> v_rows(int64_t per) const;   ///< per-value-head runs: gates (attn_gate), dt, ssm_a, alpha, beta
    std::vector<core::GdnRun> q_rows() const;     ///< QSA attn_q rows ([query | gate] pairs, head order)
    std::vector<core::GdnRun> kv_rows() const;    ///< QSA attn_k / attn_v rows (head order)

    bool owns_expert(int64_t expert) const { return expert % tp == rank; }
    int shared_owner(int64_t layer) const { return static_cast<int>(layer % tp); }
    bool owns_shared(int64_t layer) const { return shared_owner(layer) == rank; }

    ///< The rule for a manifest tensor of this layer, by bare name ("attn_q.weight", "hc_ffn_up.weight", ...).
    ///< `unknown` (when asked) marks names outside the model's roles — including the two output projections:
    ///< this plan NEVER slices a column axis, so a column request is a caller bug.
    FnPlace placement(const std::string& name, int64_t layer, bool* unknown = nullptr) const;

    ///< True when a tensor name slices a COLUMN axis in some other design — here always false, by policy:
    ///< arena column slices are not fragment-expressible (tests/core/tp_col_probe.cpp measures it).
    static bool wants_columns(const std::string& name);

    int64_t aq() const { return card.n_head * card.head_dim; }    ///< the card's query width
    int64_t kv() const { return card.n_head_kv * card.head_dim; } ///< the card's k (and v) width
};

/// Why the geometry cannot run tensor-parallel on `tp` cards; empty when it can.
std::string fn_tp_check(const core::ModelGeometry& g, int tp);

/// The loader's per-name row slices this card needs: the head-sliced mixer weights of every layer, keyed
/// `blk.<l>.<role>` (the roles whose placement is RankRows; everything else the plan places by other rules).
core::WeightTable::RowSlices fn_row_slices(const FnRankPlan& p);

/// The plan for card `rank`; `err` (non-empty) when the check fails or the rank is out of range.
FnRankPlan fn_rank_plan(const core::ModelGeometry& g, int tp, int rank, std::string& err);

}  // namespace basalt::plan
