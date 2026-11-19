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
    bool mb_open = false;                        // whether bring-up opened `mb` (so a reopen never resets it)
    // the geometries of the driver's two cards: `card0` is rank 0's own card geometry (the same object the
    // load checked rank 0's tables against - Context.fn_tp_card), `g_full` the whole model's - both Verifier
    // inits take a card geometry and read the FULL widths (the two replicated output projections) through
    // FnTp::g_full, exactly as the recorder (set_tp, 32c-2) already does.
    const basalt::core::ModelGeometry* card0 = nullptr;
    const basalt::core::ModelGeometry* g_full = nullptr;

    /// The driver's bring-up opens the pair's mailbox (imitating the dense pair's attach_tp,
    /// src/dense/stage.cpp:945-985: one mailbox for the pair, two slots - a window's exchanges alternate
    /// between them - each as wide as the payload's widest row).  `dev0` names rank 0's card; `dev` is
    /// already set with the twin.  Idempotent: the mailbox opens once per process and must not be reopened
    /// under a running window (its slot epochs live in it).
    bool open_mb(int dev0, size_t slots, size_t max_bytes, std::string& err) {
        if (mb_open) return true;
        if (!mb.open(dev0, dev, slots, max_bytes, err)) return false;
        mb_open = true;
        return true;
    }

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
