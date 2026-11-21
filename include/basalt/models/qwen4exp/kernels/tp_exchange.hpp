// include/basalt/models/qwen4exp/kernels/tp_exchange.hpp - the tensor pair's activation-first exchange: the
// place and widen steps around the mailbox.  (The exchange itself is core::TpMailbox's.)
//
// A pair card computes its model heads' activation rows; before the two REPLICATED output projections each card
// gets the whole model-head rows: `tp_place` writes the card's values at their model head slot of a full row and
// zeros the rest, TpMailbox::exchange_bf16 sums the pair's rows elementwise (x + 0 = x, rounded once - both
// cards keep bit-equal rows), `tp_bf16_widen` opens the summed row to fp32 for native_quantize_q8_1.
#pragma once

#include <cstdint>

namespace basalt::kernels {

/// Rows [0, n) of `full` (bf16, `total` values each): the card's `half` values per row (src's row stride is
/// `half`) at columns [off, off + half), zeros elsewhere; the source as fp32 (rounded here) or already bf16.
void tp_place_bf16_f32(uint16_t* full, const float* src, int64_t half, int64_t total, int64_t off, int n,
                       void* stream);
void tp_place_bf16_bf16(uint16_t* full, const uint16_t* src, int64_t half, int64_t total, int64_t off, int n,
                        void* stream);

/// THE GDN FORM of the same step.  A GDN card's rows are its value heads in PACKED order (core::gdn_head_split:
/// the cut is by key-head group, so packed head p is model head (p / nk) * h_k + k0 + p % nk - the card's rows
/// stride the model's head axis in steps of h_k, they do not sit side by side at `rank * half`).  Placing them
/// at `off = rank * half` of a replicated output projection's column axis pairs each head with ANOTHER head's
/// columns, which is the pair's whole wrong-attention-output defect: card-symmetric, first window, first layer.
/// These variants place them at their model head columns; the group strides come from
/// core::gdn_place_groups and the index law is core::gdn_grouped_src_col (ONE place, host-checkable).
void tp_place_bf16_f32_grouped(uint16_t* full, const float* src, int64_t half, int64_t total, int64_t src_group,
                               int64_t dst_group, int64_t base, int n, void* stream);
void tp_place_bf16_bf16_grouped(uint16_t* full, const uint16_t* src, int64_t half, int64_t total,
                                int64_t src_group, int64_t dst_group, int64_t base, int n, void* stream);

/// `row` (bf16, n values) widened in place to fp32 in `out` (which may be the same buffer only if it is 2x sized -
/// the pair passes distinct buffers).
void tp_bf16_widen(float* out, const uint16_t* row, int64_t n, void* stream);

}  // namespace basalt::kernels
