// src/models/qwen4exp/load/fn_tp_pair.hpp - the pair's second card, kept alive for the driver.
//
// The --tp 2 load builds rank 1's twin (its arena, its half of every sliced tensor, its native slabs) on the
// second card.  Until now the load gate freed it and refused the start; this handle holds it for the pair's
// runtime driver, which will run both halves and the mailbox between them.  The mailbox (core::TpMailbox,
// graph-safe, model-agnostic) is opened by the driver's bring-up, not here.
//
// Like the other phase resources, the handle lives on the weights() phase's stack until the phases it chains
// into return; Context.fn_tp borrows it.  Destruction order is load-safe as it stands: cudaFree is not bound
// to the current device, and the weight tables and native slabs point into memory the handle frees itself
// (the tables own nothing), so the arena outlives every pointer the members hold.
#pragma once

#include "basalt/core/tp_mailbox.hpp"
#include "basalt/core/weights.hpp"
#include "basalt/models/qwen4exp/decode/native_dense.hpp"
#include "basalt/models/qwen4exp/plan/tp_rank.hpp"

#include <cstdint>

namespace basalt::models::qwen4exp::generate {

struct FnTpPair {
    int dev = -1;                                // the pair's second card
    basalt::plan::FnRankPlan plan;               // its rank plan (plan.card: what the layout check expects)
    basalt::core::WeightTable::RowSlices slices; // its row runs on every sliced tensor
    void* arena = nullptr;                       // its weight arena, cudaMalloc'd on `dev`
    uint64_t arena_bytes = 0;
    basalt::core::WeightTable wt;                // its half of every sliced tensor; rows point into `arena`
    basalt::core::NativeDense native;            // its native projection slabs, on `dev`
    basalt::core::TpMailbox mb;                  // the pair's exchanges; the driver opens it at bring-up

    FnTpPair() = default;
    FnTpPair(const FnTpPair&) = delete;
    FnTpPair& operator=(const FnTpPair&) = delete;
    ~FnTpPair() {
        mb.close();  // self-contained: TpMailbox::close frees each card's epoch words under its own OnDevice
                     // scope and the pinned block with cudaFreeHost, which binds to no device.
        if (arena != nullptr) cudaFree(arena);
    }
};

}  // namespace basalt::models::qwen4exp::generate
