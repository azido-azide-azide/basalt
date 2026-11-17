// src/dense/internal.hpp - the dense stack's model, session state and forward passes (see basalt/dense/model.hpp).
#pragma once

#include "basalt/dense/model.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace basalt::vision {
class Mmproj;
class CudaEncoder;
}

namespace basalt::dense {

constexpr int kWindow = 16;       // rows one session's decode window holds
constexpr int kBatch = 16;        // rows a batch window (several sessions' windows) holds
constexpr int kNarrow = 8;        // the narrower MMQ tile for windows of up to 8 rows
constexpr int kDraftQ = 20;       // candidates a block drafter's position keeps at most (its selector's top_k)
constexpr int64_t kDraftVocab = 131072;   // vocabulary rows the block drafter's head scores (a multiple of 128)
constexpr int kChunk = 4096;      // tokens a prompt chunk reads at once
constexpr int kAttnSplits = 3;    // key splits the prompt attention may use: the most its plan picks for a full chunk
constexpr int kTypeF8 = 44;       // F8_E4M3 projections (Linear::type)
constexpr int kTypeQ8P = 108;     // Q8_0 in planes (window_gemm Format::Q8), made at load for the decode window
constexpr int kTypeNvfp4Planes = 45;   // NVFP4_PLANES matrices (Linear::type, nvfp4_planes.hpp)

/// Device buffers carved from one allocation.
class Arena {
public:
    ~Arena();
    bool reserve(size_t bytes);
    template <class T> T* take(size_t n) { return static_cast<T*>(take_bytes(n * sizeof(T))); }
    void* take_bytes(size_t n);
    void rewind(size_t mark) { used_ = mark; }
    size_t mark() const { return used_; }
    size_t size() const { return size_; }
private:
    uint8_t* base_ = nullptr;
    size_t size_ = 0, used_ = 0;
};

/// A batch window's device table: per window row its GDN segment, where its GDN rows go in its session, per segment its
/// first cell.
struct BatchTable {
    kernels::SegRow first[kBatch];   // per segment its first row's entry, and where its GDN rows go
    kernels::SegDst sdst[kBatch];
    int32_t pos0[kBatch];
};

/// The decode window's scratch (kBatch rows).
struct WindowScratch {
    uint16_t* x = nullptr;                       // the residual rows, BF16 (every add in fp32, rounded once)
    float *xn = nullptr, *tmp = nullptr;
    void* xq = nullptr;                          // the window gemm's activations (bf16 rows)
    void* hq = nullptr;                          // the down product's (SwiGLU's bf16 rows)
    void* x8 = nullptr;                          // MMQ's (q8_1 rows)
    kernels::wgemm::Workspace w8;                // the window gemm's split-K sums
    float* dacc = nullptr;                       // a batch window's block drafter features, packed
    float *z = nullptr;
    float *qfull = nullptr, *k = nullptr, *v = nullptr, *qo = nullptr, *ao = nullptr, *ag = nullptr;
    float *g = nullptr, *u = nullptr, *h = nullptr;
    float* attn = nullptr;                       // gqa_decode_attn's partial sums
    int32_t *tok = nullptr, *pos = nullptr;      // the window's tokens and [row][3] positions (device)
    float* cat = nullptr;                        // the drafter's [embedding | hidden] rows
    float *lp = nullptr, *lp_scratch = nullptr;  // the drafter's picks (row_logprobs rows) and their scratch
    float *y = nullptr, *hl = nullptr;           // the GDN output rows; the drafter's head rows
    int32_t *iota = nullptr, *bounds = nullptr;  // the products' row map and the window's rows
    int32_t* pos0 = nullptr;                     // the window's first cell (device), read by its attention
    // a batch window's: the packed GDN rows of the layer in hand (before they go to each session), its table
    float* rq = nullptr;
    float* gab = nullptr;                        // its GDN gates (dec_gdn_gates): alpha [rows][h_v], then beta
    BatchTable* btab = nullptr;
    // the block drafter's: its rows' q, k, v, attention and convolution coefficients, the selector's projection
    float *bq = nullptr, *bk = nullptr, *bv = nullptr, *bqo = nullptr, *bao = nullptr, *dyn = nullptr, *bh = nullptr;
    float *battn = nullptr, *blp = nullptr, *bsel = nullptr;
    int32_t* bpos = nullptr;
};

/// The prompt chunk's scratch (kChunk rows), its mixer and FFN buffers sharing memory.
struct ChunkScratch {
    float *x = nullptr, *xn = nullptr, *tmp = nullptr;
    void* xq = nullptr;                           // MMQ activations
    float *qkv = nullptr, *z = nullptr, *hc = nullptr, *gate = nullptr, *beta = nullptr, *ab = nullptr;
    float *y = nullptr, *yn = nullptr;
    void* work = nullptr;
    float *qfull = nullptr, *k = nullptr, *v = nullptr, *qo = nullptr, *ao = nullptr, *ag = nullptr;
    float* attn_split = nullptr;                  // gqa_prompt_attn's key-split scratch (kAttnSplits splits)
    float *g = nullptr, *u = nullptr, *h = nullptr;
    float *hn = nullptr, *cat = nullptr;          // the drafter's history: main's normed rows, its input rows
    float* dacc = nullptr;                        // the block drafter's projected features of the chunk's rows
    void* a4 = nullptr;                           // the FP4 activations (fp4_dense), their token scales
    float* gs = nullptr;
    float* hmax = nullptr;                        // each row's largest |H| (the SwiGLU product's), for its FP4 rounding
    int32_t *tok = nullptr, *pos = nullptr, *iota = nullptr, *bounds = nullptr;
};

struct SessionState;
/// A second card that holds the model's last layers and reads them for the main card: the prompt line reads a long prompt's
/// chunks through them while the main card reads the next chunk's first layers (stage.cpp).
class Stage;
struct StageDeleter {
    void operator()(Stage* p) const;
};
using StagePtr = std::unique_ptr<Stage, StageDeleter>;

/// The layer split's decode hand-off, page-locked and device-mapped: each card's window graph writes and reads its rows with
/// kernels and raises the other card's flag (the host lowers both before a window).
struct SplitHand {
    uint16_t* fwd = nullptr;       ///< [kBatch][n_embd] BF16: the main card's residual after its last layer
    uint16_t* ret = nullptr;       ///< [1 + taps][kBatch][n_embd] BF16: the second card's after the last layer, then after each tap
    uint32_t* flag = nullptr;      ///< [0]: fwd is in; [1]: ret is in
    std::vector<int> taps;         ///< the block drafter's target layers on the second card, in order
    size_t first_tap = 0;          ///< the first of them among the drafter's target layers
};

/// The tensor-parallel pair's window hand-off, page-locked and mapped so both cards reach it: one row block per exchange in
/// each direction, and a flag per direction counting the exchanges that card has put in.  A row-parallel product leaves each
/// card a PARTIAL of the same rows; `tp_allreduce2` gives each card the sum of the two, which is what its own shard of the
/// next matrix multiplies.  Every exchange of a window has its own block, which is what makes a card's write safe without
/// waiting for the peer to have read it: the peer's last signal of a window sits behind its read of every block of that
/// window, and this card's next window sits behind its own wait for that signal.  The flags hold the exchange's number, so
/// the waits are fixed at capture and the host lowers both flags between windows - once both cards are past the previous
/// window's last exchange, which is what keeps a raised flag of a window that has not run yet out of the next one's waits.
struct TpHand {
    uint32_t* flag = nullptr;      ///< [0]: the row blocks card 0 has put in; [1]: card 1's
    uint16_t* slot = nullptr;      ///< [2][exchanges][kBatch][n_embd] BF16: [r] is what card r wrote and the other reads
    int n_embd = 0;                ///< the width of a row block
    int exchanges = 0;             ///< the row blocks of one window: two per layer the pair runs
    void* mem = nullptr;           ///< the page-locked allocation these point into (the host view lowers the flags)
};

/// The pair's head, in one move.  The head's product writes `rows` by the matrix's own out_features CONTIGUOUS - so a
/// card's vocabulary shard leaves it a compact block, at a pitch of its own column count, wherever the product was told
/// to put it, and the logits the sampler reads are the model's whole vocabulary wide.  Assembling a row is therefore a
/// pitched copy even for the card's own half: its block into its own columns, and the peer's block - read across the
/// pair's door - into the columns that follow.  Measured on a two-card pair over a Gen4 x8 link: a flat crossing of the
/// shape this makes costs 34.7 us per window row (7.58 MiB for sixteen rows at 13.34 GiB/s), a quarter of one percent of
/// a token.  A pair whose driver refuses the door sends the peer's block through the hand-off's page-locked memory
/// instead (measured: 1195 us for the same sixteen rows, still under one percent), at the cost of one more hop the peer
/// pays - see the blueprint's head section for that shape, which tp_hand.cpp's flags already know how to do.
struct TpHead {
    float* dst = nullptr;      ///< the logits being assembled, [rows][dst_pitch]
    const float* src = nullptr;  ///< a head block: the card's own, or the peer's across the door
    int dst_pitch = 0;         ///< the logits' row stride, in values (the model's vocabulary)
    int src_pitch = 0;         ///< the block's row stride, in values (its own column count)
    int dst_col0 = 0;          ///< the column of the logits this block belongs at
    int cols = 0;              ///< the block's columns (tp_head_rows of the model's vocabulary)
};

/// the pair's door, opened once per card pair at boot: each card's kernels and copies then reach the other's memory
/// directly.  Already-open is success; a driver that refuses it is the pair's fallback, not an error.
bool tp_peer_enable(int a, int b, std::string& err);

/// `rows` of one vocabulary block into the logits, one queued copy.  The card's own block and the peer's go through the
/// same call - which half is whose is `src`'s address, which the door makes legal to read.  Queued on `s`; nothing reads
/// the logits until the sampler does, and the sampler runs after this on the same stream.
bool tp_head_place(const TpHead& h, int rows, void* s, std::string& err);

/// The pair's hand-off for `exchanges` row blocks of `n_embd` (two per layer the pair runs), page-locked, mapped, zeroed.
/// Null with `err` when the memory is not there.
TpHand* tp_hand_alloc(int n_embd, int exchanges, std::string& err);
void tp_hand_free(TpHand* h);
/// One all-reduce of `rows` x `n_embd` BF16 partials across the pair, queued on `s` as card `rank`: this card's rows into the
/// block its peer reads, its own flag raised to `e` + 1, the peer's awaited at that value, the peer's block added into this
/// card's partial.  `e` is the exchange's number in the window (`[0, TpHand::exchanges)`); both cards run the same numbers
/// in the same order, so a card's write always precedes its wait and the pair cannot wait on each other.
void tp_allreduce2(const TpHand& h, int rank, int e, uint16_t* partial, int rows, void* s);
/// The same exchange with the SUM landing added to the card's running residual rows `x` instead of back into `partial`:
/// `x` gains both cards' partials (added as floats, rounded once), which is where a row-parallel product's result belongs
/// when the next matrix it feeds is split along its inputs.  `partial` is left as it was; `x` and `partial` must differ.
void tp_allreduce2_add(const TpHand& h, int rank, int e, const uint16_t* partial, uint16_t* x, int rows, void* s);
/// Both flags lowered for the next window (both cards are past the last exchange's, as `TpHand` explains).
void tp_hand_lower(TpHand& h);

class Model {
public:
    Geometry g;                                  // the model's own widths (what the pack describes)
    Weights w;
    Options o;
    Arena dec_arena, pre_arena;
    WindowScratch ws;
    ChunkScratch cs;
    bool chunk_ready = false;
    // K/V streaming: the VRAM copy of one attention layer's far cells a prompt chunk reads (at = the resident cells; at
    // INT64_MAX without streaming), filled on stage_side from stage_free (the stage's last reader done) to stage_ready
    kernels::KvFar stage;
    cudaStream_t stage_side = nullptr;
    cudaEvent_t stage_free = nullptr, stage_ready = nullptr;
    std::unique_ptr<prefill::mmq::Context> mmq;
    int lb = 0, le = 0;                          // the layers this card runs: [lb, le)
    int n_gdn = 0, n_attn = 0;                   // the GDN and attention layers among them
    std::vector<int> gdn_index, attn_index;      // per layer: its index among the card's GDN or attention layers (-1)
    // the batch windows' graphs, one per composition (the sessions in the window and their rows), the least recently used out
    using BatchKey = std::vector<std::pair<const SessionState*, int>>;
    struct BatchGraph {
        BatchKey key;
        cudaGraphExec_t exec = nullptr;
        uint64_t used = 0;
    };
    static constexpr size_t kBatchGraphs = 64;
    std::vector<BatchGraph> batch_graphs;
    uint64_t batch_clock = 0;
    // a batch window's GDN gates run on gate_side beside the qkv/z product, from gate_fork to gate_join
    cudaStream_t gate_side = nullptr;
    cudaEvent_t gate_fork = nullptr, gate_join = nullptr;
    std::unique_ptr<vision::Mmproj> mmproj;
    std::unique_ptr<vision::CudaEncoder> encoder;
    BlockDrafter bd_dec;                         // the block drafter as decode windows run it (its products in Q8 planes)
    std::vector<void*> owned;                    // device buffers the model allocated (Q8 plane copies, a wider workspace)
    void* enc_arena = nullptr;
    size_t enc_bytes = 0;
    // The tensor-parallel pair.  Each card runs EVERY layer at `tp_cards`-th of the head and intermediate widths, so
    // everything the card sizes for itself - its scratch, its K/V cells, its delta-net state, its products' rows - comes
    // from `cg`, and the layer bodies read `cg` instead of `g` (`cg == g` with no pair).  The vocabulary keeps its full
    // width in `cg`: the head's rows split out by `tp_head_rows`, and card 0's logits hold every row once the pair
    // gathers them.  The block drafter belongs to card 0 alone (M1's placement), so only that card sizes for it.
    Geometry cg;                                 // the widths THIS card runs (== g unless it is one card of a pair)
    TpHand* tp_hand = nullptr;                   // the pair's row-block hand-off (null: no pair)
    uint16_t* tp_partial = nullptr;              // a window row block's BF16 partials awaiting their exchange
    int tp_exchange = 0;                         // the exchange number the NEXT row-parallel product of this window is

    bool owns_drafter() const { return o.tp_cards <= 1 || o.tp_rank == 0; }
    StagePtr line;                               // the prompt line: a second card reading long prompts' last layers (null: none)
    StagePtr split;                              // the layer split's second card: layers [le, n) (null: none)
    SplitHand* hand = nullptr;                   // the layer split's decode hand-off, on both cards' models (null: none)
    Model* split_model = nullptr;                // the split's second card's model, and its CUDA device
    int split_dev = -1;
    ~Model();
    bool ensure_chunk(std::string& err);
    /// llama.cpp's MMQ readied for every product of the model (its kernels loaded for each format and height).
    bool ready_products(std::string& err);
    /// The image encoder placed in device memory for pictures of up to `tokens` merged tokens.
    bool vision_attach(int tokens, std::string& err);
    /// Its memory goes back.
    void vision_detach();
};

struct SessionState;
struct PeerStateDeleter {
    void operator()(SessionState* s) const;
};
/// One conversation's device state.
struct SessionState {
    cudaStream_t stream = nullptr;
    cudaStream_t side = nullptr;                  // the GDN commit, beside the block drafter
    cudaEvent_t fork = nullptr, join = nullptr;   // the commit's start (after the window) and end
    Arena arena;
    float* gdn = nullptr;                        // per GDN layer: state [S][h_v][S], then conv history [C][3]
    // the other state buffer: a one-row window writes its state after the row there (alt_ready), and keeping the row
    // swaps the two instead of committing.  Every window graph reads the state where it was captured: wg[1] holds the
    // graphs captured while gdn was the second buffer
    float* gdn_alt = nullptr;
    float* gdn_first = nullptr;                  // the buffer gdn started as
    bool alt_ready = false;
    size_t gdn_floats = 0;                       // per layer
    std::vector<uint16_t*> kc, vc;               // per attention layer: [cell][n_kv][head_dim] fp16 (null with int8)
    std::vector<kernels::KvInt8> kq;             // per attention layer: the int8 cache ({} with fp16)
    // per attention layer, with K/V streaming: the cells from Options::kv_resident on, in page-locked host memory
    // (device-mapped); {} without.  kc/vc/kq then hold only the first kv_resident cells
    std::vector<kernels::KvFar> far_kv;
    void* far_host = nullptr;
    // the last window's rows per GDN layer, until its commit
    float *rec_qkv = nullptr, *rec_h = nullptr, *rec_gate = nullptr, *rec_beta = nullptr;
    int32_t* n_keep = nullptr;                   // device
    float* logits = nullptr;                     // [kWindow][n_vocab]
    int32_t* picks = nullptr;                    // device [kWindow]
    int64_t rope_delta = 0;                      // rope position - cell index for the next text token
    cudaGraphExec_t wg[2][kWindow + 1] = {};     // the decode window of T rows, captured once per T and state buffer
    // the drafter (null without one): its KV, the last window's normed hidden rows, and the row waiting for the
    // token after it (cell tail_cell, -1 none)
    uint16_t *mtp_kc = nullptr, *mtp_vc = nullptr;
    float *hid = nullptr, *tail = nullptr, *chain = nullptr;   // chain: [2][n_embd]
    float *dlogits = nullptr, *lp = nullptr;     // the drafter's head row and its picks [kWindow][kLogprobStride]
    int32_t* win_tok = nullptr;                  // the last window's tokens (device)
    int64_t tail_cell = -1;
    // the block drafter: its context K/V per layer (cyclic, dctx_cap cells), the last window's projected features,
    // its picks and their confidence, and the proposal each pick came from ([kWindow][top_k] candidates and their q)
    std::vector<uint16_t*> dk, dv;
    int64_t dctx_cap = 0, dctx_from = 0;         // cells; the first position its context holds
    float* dacc = nullptr;
    int32_t* dids = nullptr;
    float* dconf = nullptr;
    int32_t* dq_ids = nullptr;
    float* dq_p = nullptr;
    // how the block drafter picks (temperature 0: its best candidates; above, drawn), set by the request's sampling
    float dtemp = 0.0f;
    uint64_t dseed = 0;
    // the layer split: the second card's state of this conversation (its layers' cells and recurrent state; null: none),
    // and the device a second card's state lives on
    std::unique_ptr<SessionState, PeerStateDeleter> peer;
    int dev = -1;
};

/// One session's device state on the card of model `m` (its layers' cells and recurrent state; on the card that holds layer
/// 0 also the logits, the picks and the drafter's state), on the current device; free_state releases it there.
bool init_state(Model& m, SessionState& st, std::string& err);
void free_state(SessionState& st);
/// The layer split's second card's state of the session `st` of model `m` (st.peer).
bool split_session(Model& m, SessionState& st, std::string& err);
/// Tokens [pos, pos + n) of the session `st` across the layer split's cards, the chunks pipelined (the second card reads a
/// chunk's later layers while the main card reads the next one's first layers); `progress` and `stop` as prompt_line_read's.
/// A session with an MTP drafter reads its prompt chunk by chunk (split_chunk): that drafter reads every chunk's last rows.
bool split_read(Model& m, SessionState& st, const int32_t* tokens, int n, int64_t pos, const std::function<void(int)>& progress,
                const std::function<bool()>& stop, std::string& err);
/// A prompt chunk across the layer split's cards, one after the other (forward_chunk's arguments).
bool split_chunk(Model& m, SessionState& st, const int32_t* tokens, const float* rows, int T, int64_t pos, const int32_t* pos3,
                 std::string& err);

/// Runs a decode window of T rows at cells pos..pos+T-1 (`pos3` [T][3] their rope positions); leaves the logits in
/// st.logits and the GDN rows for the commit.
bool forward_window(Model& m, SessionState& st, const int32_t* tokens, int T, int64_t pos, const int32_t* pos3,
                    std::string& err);
/// One window of S sessions as one batch: segment k is segs[k].T rows (tokens and pos3 [row][3] packed in segment order, at
/// most kWindow rows in all) at cells segs[k].pos.. of its session.  Each session is left as after its own forward_window
/// (logits, the GDN rows for its commit, the drafter's rows); runs on the first session's stream.
struct BatchSeg {
    SessionState* st = nullptr;
    int T = 0;
    int64_t pos = 0;
};
bool forward_batch(Model& m, const BatchSeg* segs, int S, const int32_t* tokens, const int32_t* pos3, std::string& err);
/// Runs a prompt chunk; rows are token embeddings, or `rows` (fp32 [T][n_embd]) when given.  `pos3` [T][3].
bool forward_chunk(Model& m, SessionState& st, const int32_t* tokens, const float* rows, int T, int64_t pos,
                   const int32_t* pos3, std::string& err);
/// A prompt chunk in phases (forward_chunk runs them in turn).  chunk_start: the bounds (host [2], null: {0, T}) and
/// positions, then the input rows into the BF16 residual (token embeddings, or `rows`; neither: the caller places it).
void chunk_start(Model& m, SessionState& st, const int32_t* tokens, const float* rows, int T, const int32_t* pos3,
                 const int32_t* bounds = nullptr);
/// Layers [l0, l1) of the chunk in m.cs; with `taps`, the block drafter's features of its target layers in the range
/// into cs.dacc (the first target layer's written, the later ones' added).  The last layer's rows are made in full when
/// something reads them (`read_last`, a drafter), else only its cells.
void chunk_layers(Model& m, SessionState& st, int T, int64_t pos, int l0, int l1, const BlockDrafter* taps, bool read_last = false);
/// The block drafter's feature projection j of the rows in cs.x into cs.dacc (j == 0 writes it, later ones add).
void chunk_tap(Model& m, SessionState& st, int T, const BlockDrafter& taps, size_t j);
/// After the last layer: the block drafter's context from cs.dacc, or the MTP history.
bool chunk_finish(Model& m, SessionState& st, int T, int64_t pos, std::string& err);
/// The block drafter's context cells at positions pos .. pos + n - 1 from n rows of features `dacc` (device).
void block_ctx_chunk(Model& m, SessionState& st, const float* dacc, int n, int64_t pos);

/// The line on CUDA device `dev` with layers [l_min, n_layers) of `w` (their weights already on `dev`, the other layers
/// empty); both cards' prompt rates are timed here.
StagePtr make_prompt_line(Model& main, int dev, int l_min, const Weights& w, std::string& err);
/// Reads tokens [pos, pos + n) of the session `st` (of model `main`) across the cards when the rates price that cheaper,
/// true; false with empty `err` when it does not (the caller reads them on the main card).
/// `stop` is asked between chunks on both cards: true ends the read where it is (the session then holds a part of the prompt
/// only, as after a stopped read on the main card), true returned.
bool prompt_line_read(Stage& line, Model& main, SessionState& st, const int32_t* tokens, int n, int64_t pos,
                      const std::function<void(int)>& progress, const std::function<bool()>& stop, std::string& err);

/// The commit of the last window's first n rows into the recurrent state.
void commit_window(Model& m, SessionState& st, int n, cudaStream_t s);

/// The block drafter's context from the last window's first n rows (st.dacc) at positions pos.. .
bool block_ctx_window(Model& m, SessionState& st, int n, int64_t pos, std::string& err);
/// The block drafter's pass after `anchor` at `pos`: its picks for slots 1 .. block-1 into st.dids / st.dconf.
/// The block is `width` rows (the drafter's block or wider, up to kWindow: one anchor, then mask tokens).
bool block_draft(Model& m, SessionState& st, int32_t anchor, int64_t pos, int width, std::string& err);
/// The same pass for S sessions at once (session k's block of widths[k] rows, at least the drafter's block, at most kBatch rows
/// in all), on `stream` (null: the first session's).
bool block_draft_batch(Model& m, SessionState* const* sts, const int32_t* anchors, const int64_t* pos, const int* widths, int S,
                       std::string& err, cudaStream_t stream = nullptr);

/// The drafter over n rows of (hidden, next token) pairs at cells c0..c0+n-1: `hid` device rows [n][n_embd] (main's
/// normed hidden, or the drafter's own output), `tok` device ids of the tokens after them.  Writes its KV; with
/// `out` (device [n][n_embd]) also its normed output rows and, for the last row, the head's pick and probability
/// into lp_row (row_logprobs layout, device).  n <= kWindow.
bool mtp_window(Model& m, SessionState& st, const float* hid, const int32_t* tok, int n, int64_t c0, float* out,
                float* lp_row, std::string& err);
/// The drafter's history over a prompt chunk's pairs (n <= kChunk), KV only: hidden rows `hid`, next tokens `tok`
/// (device ids) or next rows `emb` (device [n][n_embd]).
bool mtp_chunk(Model& m, SessionState& st, const float* hid, const int32_t* tok, const float* emb, int n, int64_t c0,
               std::string& err);

}  // namespace basalt::dense
