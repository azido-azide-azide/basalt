// include/basalt/dense/weight_shard.hpp - one weight matrix as one card of a tensor-parallel pair holds it: the runs
// of one dimension the card owns, and the byte spans that lay the card's copy out of the pack's tensor.
//
// The card's copy keeps the pack's layout - it is a whole matrix of the card's own shape - so every product reads it
// the way it reads the unsplit one and its output is the product's partial sum (a cut along the input dimension) or
// its share of the outputs (a cut along the output dimension).  What the cut must leave whole is the type's block
// along K: 32 values for Q8_0, one for F32, BF16 and F8_E4M3, and - for NVFP4 planes - the 64-value group the scale
// tiles cover (a tile is 128 rows x 4 groups, row-block major), whose rows the cut along the outputs also takes in
// whole 128-row blocks.  `shard_spans` says where a cut does not.
//
// Free of CUDA: a CPU test builds the plan for a whole pack and checks the spans against the pack's own tensor table.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::dense {

/// The dimension a cut takes apart: a matrix's output features, its input features, or neither (the card holds it whole).
enum class ShardDim { None, Rows, Cols };

/// `count` elements of the cut dimension at `first`; a card's runs are packed into its copy in this order.
struct ShardRun {
    int64_t first = 0, count = 0;
};

/// One matrix of `rows` x `cols` as one card holds it: `runs` are its part of `dim`, the other dimension whole.
struct MatrixShard {
    ShardDim dim = ShardDim::None;
    int64_t rows = 0, cols = 0;
    std::vector<ShardRun> runs;

    bool whole() const { return dim == ShardDim::None; }
    int64_t run_values() const;    ///< the card's values of the cut dimension
    int64_t shard_rows() const;    ///< the card's copy's rows
    int64_t shard_cols() const;    ///< the card's copy's cols
};

/// A run of bytes: `n` of the whole tensor at `src` to the card's copy of it at `dst`.
struct ByteSpan {
    int64_t src = 0, dst = 0, n = 0;
};

/// The spans that lay the card's copy out of the whole tensor's bytes, `type` a ggml id (F32 0, Q8_0 8, BF16 30,
/// F8_E4M3 44, NVFP4_PLANES 45).  False with `err` when the cut leaves a block or a scale tile in two cards' hands,
/// or the type is not one a matrix shard reads.
bool shard_spans(uint32_t type, const MatrixShard& shard, std::vector<ByteSpan>& out, std::string& err);

/// The bytes the card's copy takes: a whole matrix of the card's own shape.  0 with `err` for a type or a shape the
/// shard does not read.
int64_t shard_bytes(uint32_t type, const MatrixShard& shard, std::string& err);

/// `spans` from the whole tensor's bytes into the card's copy, which must hold `shard_bytes`.
void gather_spans(const uint8_t* full, const std::vector<ByteSpan>& spans, uint8_t* out);

/// The card `rank` of `width` holds as one run: whole units of `unit` values, the `n_units` of the dimension shared
/// out evenly, in the pack's order.  Empty when `n_units` does not divide.
std::vector<ShardRun> unit_runs(int64_t n_units, int64_t unit, int width, int rank);

/// `runs` moved on by `offset` and appended to `to` (a tensor that stacks two axes of the same units).
void append_runs(std::vector<ShardRun>& to, const std::vector<ShardRun>& runs, int64_t offset);

}  // namespace basalt::dense
