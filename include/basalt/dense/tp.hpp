// include/basalt/dense/tp.hpp - the tensor-parallel plan for the dense stack: the geometry one card of a tp-card group
// holds, the weight slices of every projection that card reads, and the byte fragments of a pack tensor those slices
// take (so a load reads only its own slice, repacked into the card's row/column order).
//
// Row-parallel projections (attention q/k/v, GDN qkv/z, FFN gate/up, the GDN per-head scalars and conv) split by
// OUTPUT rows: each card runs its head/channel slice over the full activation and gets disjoint outputs.
// Column-parallel ones (attention o, GDN out, FFN down) split by K: each card reads its column slice of the card-
// local activations and writes a partial sum of the same outputs; the sums meet over the peer mailbox
// (basalt/core/tp_mailbox.hpp).
//
// Head halves: the attention card takes query heads [r·n_head/tp, ...) and key-value heads [r·n_kv/tp, ...) (the
// grouping is preserved); the GDN card takes key heads [r·k/tp, ...) and their value heads - exactly the head slice
// of basalt/core/gdn_head_split.hpp, so the sliced GDN layer is a GDN layer of k/tp key heads and v/tp value heads
// with the same "value head h meets key head h % k_heads" rule.  The FFN splits its intermediate units contiguously.
// The head, the token embedding, the final norm and the drafter stay on rank 0 (the card that samples); a later
// slice puts them across the cards too.
//
// tp is 2 for now; every rule here states itself for a general divisor.
#pragma once

#include "basalt/core/gdn_head_split.hpp"
#include "basalt/dense/model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::dense::tp {

/// One card's plan: its geometry (heads/FFN/GDN widths divided by tp) and the element runs of each weight class,
/// in pack order -> card order.
struct RankPlan {
    int tp = 1, rank = 0;
    Geometry geom;            ///< this card's geometry: the kernels and the session see only this
    core::GdnHeadSlice gdn;   ///< the card's key-head range of the full GDN layer

    ///< the card's key heads and value heads (for the session's head map: dec_gdn_rec a.k0 / a.nk)
    int k0() const { return gdn.k0; }
    int nk() const { return gdn.nk; }
    int nv() const { return gdn.nv(); }

    // ---- row-parallel classes: runs over the tensor's ROW axis (rows of any width) ----
    std::vector<core::GdnRun> qkv_rows() const;   ///< GDN qkv output rows and the conv's channels (the qkv channel axis)
    std::vector<core::GdnRun> v_rows(int64_t per) const;  ///< per-value-head rows/columns of `per` values: z, dt, ssm_a, alpha, beta
    std::vector<core::GdnRun> q_rows() const;     ///< attention q rows ([query | gate] pairs, in head order)
    std::vector<core::GdnRun> kv_rows() const;    ///< attention k and v rows (head order)
    std::vector<core::GdnRun> ffn_rows() const;   ///< FFN gate/up rows (intermediate units)

    // ---- column-parallel classes: runs over the tensor's COLUMN (K) axis ----
    std::vector<core::GdnRun> o_cols() const;     ///< attention output columns (per query head)
    std::vector<core::GdnRun> out_cols() const;   ///< GDN out columns = the card's value-head columns
    std::vector<core::GdnRun> ffn_cols() const;   ///< FFN down columns (intermediate units)

    ///< the card's q/k/v widths in rows (2·AQ includes the gate rows).
    int64_t aq() const { return (int64_t) geom.n_head * geom.head_dim; }
    int64_t kv() const { return (int64_t) geom.n_kv * geom.head_dim; }
};

/// Why `g` cannot run tensor-parallel on `tp` cards; empty when it can.  Checks the divisibility every rule above
/// needs and the block alignment the quantized slices take (heads a whole multiple of tp, the per-card geometry
/// passing Geometry::check's shapes for the kernels that read a card's slice).
std::string check(const Geometry& g, int tp);

/// The plan for card `rank`; `err` (non-empty) when `check` fails or rank is out of range.
RankPlan rank_plan(const Geometry& g, int tp, int rank, std::string& err);

// ---- pack bytes: which fragments of a tensor's payload a slice is ------------------------------------------

/// The GGUF tensor types a dense card slice reads.
enum : uint32_t { kF32 = 0, kQ8_0 = 8, kBf16 = 30, kNvfp4Planes = 45 };

/// A byte fragment of a payload: `len` bytes at `src` in the full tensor's payload go to `dst` in the slice's.
/// Fragments come back in ascending `dst` order.
struct Frag {
    uint64_t src, dst, len;
};

/// The fragments of a `rows` x `cols` tensor of `type` taking the element runs `runs` of one axis (the row axis
/// when `col_axis` is false, the column/K axis when true).  Runs are in values of that axis.  A type's block
/// alignment is required and checked: Q8_0 runs on multiples of 32 values, NVFP4_PLANES column runs on multiples of
/// 64 (whole 4-group scale tiles) with rows a multiple of 128, NVFP4_PLANES row runs on multiples of 128 rows;
/// F32 and BF16 take any runs.  `rows`/`cols` are the FULL tensor's; the slice's are `runs` totals.  Empty when
/// the type is unknown or a run misaligned (the caller errors).
std::vector<Frag> payload_fragments(uint32_t type, int64_t rows, int64_t cols, const std::vector<core::GdnRun>& runs,
                                    bool col_axis);

/// The bytes a slice takes: the fragments' lengths summed.
uint64_t shard_bytes(const std::vector<Frag>& frags);

/// The slice's payload from the full one (distinct buffers; `slice` must be `shard_bytes` long).
void gather_payload(const void* full, const std::vector<Frag>& frags, void* slice);

}  // namespace basalt::dense::tp
