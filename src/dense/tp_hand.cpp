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

TpHand* tp_hand_alloc(int n_embd, int exchanges, std::string& err, int chunk_rows) {
    if (n_embd <= 0 || exchanges <= 0 || chunk_rows < 0) {
        err = "dense: a tensor-parallel hand-off of " + std::to_string(exchanges) + " row blocks of " + std::to_string(n_embd);
        return nullptr;
    }
    TpHand* h = new TpHand();
    h->n_embd = n_embd;
    h->exchanges = exchanges;
    const size_t block = (size_t) kBatch * (size_t) n_embd * 2;   // one window row block, BF16
    const size_t cbytes = chunk_rows > 0 ? 2 * (size_t) chunk_rows * (size_t) n_embd * 2 : 0;
    const size_t bytes = 256 + 2 * (size_t) exchanges * block + cbytes;   // flags, window blocks, prompt block
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
    h->read = h->flag + 2;   // the prompt block's other half of the handshake: what the peer has taken out of it
    h->slot = (uint16_t*) (at + 256);
    h->chunk_rows = chunk_rows;
    h->chunk = chunk_rows > 0 ? h->slot + 2 * (size_t) exchanges * kBatch * (size_t) n_embd : nullptr;
    if (chunk_rows > 0)
        basalt::log::info("basalt dense",
                          "tensor parallel: %.1f MiB of page-locked hand-off, %d row blocks of %d each way, and a %d-row "
                          "prompt block each way", (double) bytes / 1048576.0, exchanges, n_embd, chunk_rows);
    else
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
    // the bytes, as floats, written THROUGH this card's caches: a slot the peer reads must hold them by the time the
    // kernel that put them is done, and a plain store only promises they left the thread (see copy_to_mapped_wt)
    kernels::copy_to_mapped_wt((float*) tp_block(h, rank, e), (const float*) partial, values / 2, s);
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

void tp_meet(const TpHand& h, int rank, int e, void* s) {
    tp_check(h, e, 1);   // the same range rule as an exchange's number; no rows are involved
    kernels::signal_flag(h.flag + rank, (uint32_t) e + 1, s);
    kernels::wait_flag_ge(h.flag + (1 - rank), (uint32_t) e + 1, s);
}

void tp_allreduce2_add_grouped(const TpHand& h, int rank, int& seq, const uint16_t* partial, uint16_t* x, int rows,
                               int group, void* s) {
    if (h.chunk_rows <= 0 || group <= 0 || group > h.chunk_rows || rows <= 0) {
        basalt::log::error("basalt",
                           "dense: a grouped tensor-parallel exchange of %d rows in groups of %d does not fit a %d-row "
                           "prompt block", rows, group, h.chunk_rows);
        std::exit(1);
    }
    uint16_t* mine = h.chunk + (size_t) rank * h.chunk_rows * h.n_embd;          // the block my peer waits on
    uint16_t* peers = h.chunk + (size_t) (1 - rank) * h.chunk_rows * h.n_embd;   // the block I read
    for (int at = 0; at < rows; at += group) {
        const int n = rows - at < group ? rows - at : group;
        const int64_t values = (int64_t) n * h.n_embd;
        ++seq;   // one number per group, running through the chunk: no wait can be answered by an older product's flag
        if (seq > 1) kernels::wait_flag_ge(h.read + (1 - rank), (uint32_t) (seq - 1), s);   // peer drained the block first
        // the window's two kernels, with the two disciplines M4 cost a hardware afternoon to learn: the bytes go THROUGH
        // this card's caches (copy_to_mapped_wt) and the peer's are read with .cv
        kernels::copy_to_mapped_wt((float*) mine, (const float*) (partial + (size_t) at * h.n_embd), values / 2, s);
        kernels::signal_flag(h.flag + rank, (uint32_t) seq, s);
        kernels::wait_flag_ge(h.flag + (1 - rank), (uint32_t) seq, s);
        kernels::add_pair_bf16(x + (size_t) at * h.n_embd, partial + (size_t) at * h.n_embd, peers, values, s);
        kernels::signal_flag(h.read + rank, (uint32_t) seq, s);   // done with the group its peer must now refill
    }
}

bool tp_peer_enable(int a, int b, std::string& err) {
    if (a == b) {
        err = "dense: a pair needs two different cards";
        return false;
    }
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, a, b) != cudaSuccess || !can) {
        err = "dense: CUDA" + std::to_string(a) + " cannot reach CUDA" + std::to_string(b) + " directly";
        return false;
    }
    if (cudaSetDevice(a) != cudaSuccess) {
        err = "dense: cannot select CUDA" + std::to_string(a) + " to open the pair's door";
        return false;
    }
    const cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
    (void) cudaSetDevice(0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
        (void) cudaGetLastError();
        err = "dense: CUDA" + std::to_string(a) + " cannot reach CUDA" + std::to_string(b) + " directly: " + cudaGetErrorString(e);
        return false;
    }
    (void) cudaGetLastError();   // an already-open door leaves the error to clear, like any refused call
    return true;
}

bool tp_head_place(const TpHead& h, int rows, void* s, std::string& err) {
    // A block that would write past the row, or read past its own, is refused rather than handed to the driver: the
    // vocabulary's width comes from the pack, so getting this wrong is a wrong number somewhere upstream, and the
    // sampler reading a row that was never filled is the kind of failure that shows up as a strange token weeks later.
    if (rows <= 0 || h.cols <= 0 || h.dst_col0 < 0 || !h.dst || !h.src || h.src_pitch < h.cols || h.dst_pitch < h.cols ||
        h.dst_col0 + h.cols > h.dst_pitch) {
        err = "dense: a head block of " + std::to_string(rows) + " rows x " + std::to_string(h.cols) + " columns at column " +
              std::to_string(h.dst_col0) + " of a " + std::to_string(h.dst_pitch) + "-wide row does not fit the logits";
        return false;
    }
    const cudaError_t e =
        cudaMemcpy2DAsync(h.dst + h.dst_col0, (size_t) h.dst_pitch * sizeof(float), h.src, (size_t) h.src_pitch * sizeof(float),
                          (size_t) h.cols * sizeof(float), (size_t) rows, cudaMemcpyDeviceToDevice, (cudaStream_t) s);
    if (e != cudaSuccess) {
        err = std::string("dense: the head block copy failed: ") + cudaGetErrorString(e);
        (void) cudaGetLastError();
        return false;
    }
    return true;
}

void tp_hand_lower(TpHand& h) {
    volatile uint32_t* f = h.flag;
    f[0] = 0;
    f[1] = 0;
    f[2] = 0;   // the prompt block's two counters too: they count groups, and a chunk starts them again from zero
    f[3] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

}  // namespace basalt::dense
