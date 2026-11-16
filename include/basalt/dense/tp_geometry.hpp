// include/basalt/dense/tp_geometry.hpp - the widths one card of a tensor-parallel pair runs: the heads it owns and
// everything that follows from them.
//
// A pair shares the model's heads out, whole heads only (the shard rules in the model's own loader say which heads a
// tensor's card holds and in what order).  What follows is that each card runs the same layer code as a smaller model:
// its query heads with the gates stacked behind them, their key/value heads, its key-head group of the delta-net -
// which core/gdn_head_split.hpp packs as a delta-net layer of its own - and its part of the feed-forward's intermediate
// width.  So the card's `Geometry` is the model's with those counts replaced, and the layer code reads its widths from
// it unchanged; the embedding, the head's per-row scales, n_embd, the head width, the delta-net's head size and the
// norms are the model's, and what a row-parallel product leaves the card as a partial sum of the FULL width is put back
// by the pair's exchange (internal.hpp's TpHand).
//
// The head is the one width whose card's share is not a width of the geometry: it is the head Linear's own rows, and the
// logits a card scores are those rows.
//
// Free of CUDA: the split plan and the host tests price a pair from this without a device.
#pragma once

#include "basalt/core/gdn_head_split.hpp"
#include "basalt/dense/model.hpp"

#include <string>

namespace basalt::dense {

/// `g` as card `rank` of a pair (or wider) of `width` cards, in `out`; `out` is `g` itself when `width` is 1.  False with
/// `err` when a width does not share out into whole heads or whole units, which is what makes a card's copy of every
/// matrix a matrix of its own shape (the loader refuses the cut for the same reason, per tensor).
inline bool tp_card_geometry(const Geometry& g, int width, int rank, Geometry& out, std::string& err) {
    out = g;
    const int cards = width < 1 ? 1 : width;   // a width below one is the model on one card
    if (rank < 0 || rank >= cards) {
        err = "dense: card " + std::to_string(rank) + " is not one of " + std::to_string(cards);
        return false;
    }
    if (cards <= 1) return true;
    const auto bad = [&](bool off, const char* what) {
        if (!off) return false;
        err = "dense: the model's " + std::string(what) + " do not share out between " + std::to_string(cards) + " cards";
        return true;
    };
    if (bad(g.n_head % cards != 0, "query heads")) return false;
    if (bad(g.n_kv % cards != 0, "key-value heads")) return false;
    if (bad(g.gdn_k_heads % cards != 0, "delta-net key heads")) return false;
    if (bad(g.gdn_v_heads % g.gdn_k_heads != 0, "the delta-net's value-to-key head ratio")) return false;
    if (bad(g.n_ff % cards != 0, "the feed-forward's intermediate width")) return false;
    if (bad(g.head_dim <= 0 || g.gdn_state <= 0, "the head width and delta-net head size")) return false;
    // The key-head group the card owns, in gdn_head_split.hpp's packed order: nk key heads and the value heads that read
    // them.  Packed so, the card's delta-net is a delta-net layer of those counts, which is what the card's geometry says.
    const int nk = g.gdn_k_heads / cards;
    const core::GdnHeadSlice slice{g.gdn_k_heads, g.gdn_v_heads, g.gdn_state, rank * nk, nk};
    out.n_head = g.n_head / cards;
    out.n_kv = g.n_kv / cards;
    out.gdn_k_heads = slice.nk;
    out.gdn_v_heads = slice.nv();
    out.n_ff = g.n_ff / cards;
    return true;
}

/// The key-head group card `rank` of `width` runs (its packed widths are the geometry's `conv_channels` and
/// `value_width`; the group is what a host test checks the loader's rows against).
inline core::GdnHeadSlice tp_gdn_group(const Geometry& g, int width, int rank) {
    const int nk = width <= 1 ? g.gdn_k_heads : g.gdn_k_heads / width;
    return {g.gdn_k_heads, g.gdn_v_heads, g.gdn_state, width <= 1 ? 0 : rank * nk, nk};
}

/// The floats one card's delta-net layer holds: its state, then its convolution history (`hist` rows per channel).
inline size_t tp_gdn_state_floats(const Geometry& g, int width, int rank, int hist) {
    return core::gdn_slice_state_floats(tp_gdn_group(g, width, rank), hist);
}

/// The head's rows card `rank` of `width` scores (the head's vocabulary shared out in whole rows; the pack's per-row
/// scales share out the same way).  0 when the vocabulary does not divide.
inline int64_t tp_head_rows(const Geometry& g, int width, int /*rank*/) {
    if (width <= 1) return g.n_vocab;
    if (g.n_vocab % width != 0) return 0;
    return g.n_vocab / width;
}

}  // namespace basalt::dense
