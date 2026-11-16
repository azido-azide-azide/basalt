// src/dense/tp_hand.cpp - the tensor-parallel pair's window hand-off (see src/dense/internal.hpp): the page-locked memory
// both cards see, the four kernels one all-reduce costs, and the flags the host lowers between windows.
#include "internal.hpp"

#include "basalt/kernels/elementwise.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace basalt::dense {

TpHand* tp_hand_alloc(int n_embd, int exchanges, std::string& err) {
    if (n_embd <= 0 || exchanges <= 0) {
        err = "dense: a tensor-parallel hand-off of " + std::to_string(exchanges) + " row blocks of " + std::to_string(n_embd);
        return nullptr;
    }
    TpHand* h = new TpHand();
    h->n_embd = n_embd;
    h->exchanges = exchanges;
    const size_t block = (size_t) kBatch * (size_t) n_embd * 2;   // one window row block, BF16
    const size_t bytes = 256 + 2 * (size_t) exchanges * block;    // the flags, then both directions
    void* mem = nullptr;
    if (cudaHostAlloc(&mem, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        delete h;
        err = "dense: out of page-locked memory for the tensor-parallel pair's hand-off";
        return nullptr;
    }
    std::memset(mem, 0, bytes);
    void* dptr = nullptr;
    if (cudaHostGetDevicePointer(&dptr, mem, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFreeHost(mem);
        delete h;
        err = "dense: the tensor-parallel pair's hand-off is not reachable from the card";
        return nullptr;
    }
    h->mem = mem;
    uint8_t* at = (uint8_t*) dptr;   // one address space: both cards read and write these pages
    h->flag = (uint32_t*) at;
    h->slot = (uint16_t*) (at + 256);
    basalt::log::info("basalt dense", "tensor parallel: %.1f MiB of page-locked hand-off, %d row blocks of %d each way",
                      (double) bytes / 1048576.0, exchanges, n_embd);
    return h;
}

void tp_hand_free(TpHand* h) {
    if (!h) return;
    if (h->mem) (void) cudaFreeHost(h->mem);
    delete h;
}

namespace {

/// the block of exchange `e` that card `wrote` filled and the other one reads
uint16_t* tp_block(const TpHand& h, int wrote, int e) {
    return h.slot + ((size_t) wrote * h.exchanges + (size_t) e) * (size_t) kBatch * h.n_embd;
}

/// both forms of the all-reduce take the same window row block and the same exchange number
void tp_check(const TpHand& h, int e, int rows) {
    if (rows <= 0 || rows > kBatch || e < 0 || e >= h.exchanges) {
        basalt::log::error("basalt", "dense: a tensor-parallel exchange of %d rows at number %d of %d does not fit the hand-off",
                           rows, e, h.exchanges);
        std::exit(1);
    }
}

/// card `rank`'s half of exchange `e`: its rows into the block its peer reads, its own flag raised to `e` + 1, then the
/// wait for the peer's flag at that value.  Everything queued after it on `s` reads the peer's block of `e`.
void tp_send_wait(const TpHand& h, int rank, int e, const uint16_t* partial, int64_t values, void* s) {
    kernels::copy_from_mapped((float*) tp_block(h, rank, e), (const float*) partial, values / 2, s);   // the bytes, as floats
    kernels::signal_flag(h.flag + rank, (uint32_t) e + 1, s);
    kernels::wait_flag_ge(h.flag + (1 - rank), (uint32_t) e + 1, s);
}

}  // namespace

void tp_allreduce2(const TpHand& h, int rank, int e, uint16_t* partial, int rows, void* s) {
    tp_check(h, e, rows);
    const int64_t values = (int64_t) rows * h.n_embd;   // the row block's BF16 values
    tp_send_wait(h, rank, e, partial, values, s);
    kernels::add_from_mapped_bf16(partial, tp_block(h, 1 - rank, e), values, s);
}

void tp_allreduce2_add(const TpHand& h, int rank, int e, const uint16_t* partial, uint16_t* x, int rows, void* s) {
    tp_check(h, e, rows);
    const int64_t values = (int64_t) rows * h.n_embd;
    tp_send_wait(h, rank, e, partial, values, s);
    kernels::add_pair_bf16(x, partial, tp_block(h, 1 - rank, e), values, s);   // the residual gains both partials
}

void tp_hand_lower(TpHand& h) {
    volatile uint32_t* f = h.flag;
    f[0] = 0;
    f[1] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

}  // namespace basalt::dense
