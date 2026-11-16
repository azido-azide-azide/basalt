// tests/models/qwen35/tp_shard_test.cpp - the tensor-parallel split of a qwen35 pack.  The model's heads share out
// evenly, every tensor of the pack has a rule, and the two cards' byte spans cut a tensor in two: a byte to one card
// only, none left out, each copy a whole matrix of its own shape.  The delta-net's slice is then checked against the
// layout the kernels read - a card's NVFP4 scale byte for its row and group is the whole matrix's byte for that head's
// row, inside the 512-byte tiles - and so is the head's per-row scale.  Host only; the pack's tensor table and bytes
// need a pack (argument or BASALT_QWEN35_PACK), and without one the plan's own arithmetic still runs and it exits 77.
#include "models/qwen35/meta.hpp"
#include "models/qwen35/shard.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/dense/weight_shard.hpp"
#include "basalt/kernels/nvfp4_planes.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace basalt;
using basalt::dense::ByteSpan;
using basalt::dense::MatrixShard;
using basalt::dense::ShardDim;
using basalt::dense::ShardRun;
using namespace basalt::models::qwen35;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// the whole dimension the pair's runs cut: every element on one card, in the pack's order
bool partitions(const std::vector<ShardRun>& a, const std::vector<ShardRun>& b, int64_t dim) {
    std::vector<std::pair<int64_t, int64_t>> spans;
    for (const std::vector<ShardRun>* v : {&a, &b})
        for (const ShardRun& r : *v) spans.push_back({r.first, r.first + r.count});
    std::sort(spans.begin(), spans.end());
    int64_t at = 0;
    for (const auto& [lo, hi] : spans) {
        if (lo != at) return false;
        at = hi;
    }
    return at == dim;
}

// the whole element a card's copy holds at its own index `at`
int64_t global_at(const std::vector<ShardRun>& runs, int64_t at) {
    for (const ShardRun& r : runs) {
        if (at < r.count) return r.first + at;
        at -= r.count;
    }
    return -1;
}

// a pack tensor, as the loader reads it: `shape[0]` is a row's width
bool shape_of(const GgufFile& gg, const std::string& name, int64_t& rows, int64_t& cols) {
    const TensorInfo* t = gg.find(name);
    if (!t) return false;
    cols = t->shape.size() > 1 ? (int64_t) t->shape[0] : 1;
    rows = (int64_t) t->elements() / cols;
    return true;
}

MatrixShard shard_of(const GgufFile& gg, const dense::Geometry& g, const std::string& name, int rank) {
    MatrixShard s;
    std::string err;
    int64_t rows = 0, cols = 0;
    if (!shape_of(gg, name, rows, cols)) {
        check(false, name + " is in the pack");
        return s;
    }
    if (!tp_shard(g, name, rows, cols, 2, rank, s, err)) check(false, name + ": " + err);
    return s;
}

// both cards' spans over one tensor: inside it, no byte on two cards, no byte on none, each copy written whole
bool tiles(const TensorInfo& t, const MatrixShard s[2], std::string& err) {
    int64_t bytes[2] = {0, 0};
    std::vector<std::pair<int64_t, int64_t>> whole;
    for (int rank = 0; rank < 2; ++rank) {
        err.clear();
        bytes[rank] = dense::shard_bytes(t.type, s[rank], err);
        if (bytes[rank] <= 0) {
            err = t.name + ": " + err;
            return false;
        }
        std::vector<ByteSpan> spans;
        if (!dense::shard_spans(t.type, s[rank], spans, err)) {
            err = t.name + ": " + err;
            return false;
        }
        int64_t sum = 0;
        std::vector<std::pair<int64_t, int64_t>> copy;
        for (const ByteSpan& p : spans) {
            if (p.n <= 0 || p.src < 0 || p.dst < 0 || p.src + p.n > (int64_t) tensor_payload_bytes(t) || p.dst + p.n > bytes[rank])
                return (err = t.name + ": a span outside the tensor or the copy", false);
            sum += p.n;
            whole.push_back({p.src, p.src + p.n});
            copy.push_back({p.dst, p.dst + p.n});
        }
        if (sum != bytes[rank]) return (err = t.name + ": its spans hold " + std::to_string(sum) + " of " +
                                        std::to_string(bytes[rank]) + " bytes", false);
        std::sort(copy.begin(), copy.end());
        int64_t at = 0;
        for (const auto& [lo, hi] : copy) {
            if (lo != at) return (err = t.name + ": its copy has a gap or an overlap", false);
            at = hi;
        }
    }
    if (bytes[0] + bytes[1] != (int64_t) tensor_payload_bytes(t))
        return (err = t.name + ": the cards hold " + std::to_string(bytes[0] + bytes[1]) + " of " +
                std::to_string(tensor_payload_bytes(t)) + " bytes", false);
    std::sort(whole.begin(), whole.end());
    int64_t at = 0;
    for (const auto& [lo, hi] : whole) {
        if (lo != at) return (err = t.name + ": " + std::to_string(lo - at) + " bytes fall between the cards", false);
        at = hi;
    }
    if (at != (int64_t) tensor_payload_bytes(t))
        return (err = t.name + ": its spans stop at " + std::to_string(at), false);
    return true;
}

// an NVFP4 planes matrix on one card: its codes and its scale bytes are the whole matrix's for the values it owns
void planes_match(const GgufFile& gg, const dense::Geometry& g, const std::string& name, int rank, int64_t row_step,
                  int64_t col_step) {
    const TensorInfo* t = gg.find(name);
    if (!t || t->type != kTypeNvfp4Planes) return check(false, name + " is in the pack as NVFP4 planes");
    const MatrixShard s = shard_of(gg, g, name, rank);
    std::string err;
    const int64_t bytes = dense::shard_bytes(t->type, s, err);
    if (bytes <= 0) return check(false, name + ": " + err);
    std::vector<ByteSpan> spans;
    if (!dense::shard_spans(t->type, s, spans, err)) return check(false, name + ": " + err);
    std::vector<uint8_t> card((size_t) bytes);
    dense::gather_spans(gg.tensor_data(*t), spans, card.data());

    const int64_t r = s.rows, k = s.cols, rc = s.shard_rows(), kc = s.shard_cols();
    const bool by_rows = s.dim == ShardDim::Rows;
    const uint8_t* full = gg.tensor_data(*t);
    const int64_t scales_full = (int64_t) kernels::nvfp4p::scale_offset(r, k);
    const int64_t scales_card = (int64_t) kernels::nvfp4p::scale_offset(rc, kc);
    bool codes = true, scales = true, rows_ok = true;
    for (int64_t i = 0; i < rc && codes && scales; i += row_step) {
        const int64_t gi = by_rows ? global_at(s.runs, i) : i;
        rows_ok = rows_ok && gi >= 0;
        for (int64_t j = 0; j < kc; j += col_step) {
            const int64_t gj = by_rows ? j : global_at(s.runs, j);
            if (card[(size_t) i * (kc / 2) + j / 2] != full[(size_t) gi * (k / 2) + gj / 2]) codes = false;
            const uint8_t sc = card[(size_t) scales_card + (size_t) kernels::nvfp4p::scale_index(i, j / 16, kc)];
            const uint8_t sf = full[(size_t) scales_full + (size_t) kernels::nvfp4p::scale_index(gi, gj / 16, k)];
            if (sc != sf) scales = false;
        }
    }
    check(rows_ok && codes, name + " (card " + std::to_string(rank) + "): its codes are the values its heads own");
    check(rows_ok && scales, name + " (card " + std::to_string(rank) + "): its scale tiles are the values its heads own");
}

// the head's per-row scales, and a delta-net's convolution rows, both plain floats on a row split
void floats_match(const GgufFile& gg, const dense::Geometry& g, const std::string& name, int rank) {
    const TensorInfo* t = gg.find(name);
    if (!t || t->type != kTypeF32) return check(false, name + " is in the pack as F32");
    const MatrixShard s = shard_of(gg, g, name, rank);
    std::string err;
    const int64_t bytes = dense::shard_bytes(t->type, s, err);
    if (bytes <= 0) return check(false, name + ": " + err);
    std::vector<ByteSpan> spans;
    if (!dense::shard_spans(t->type, s, spans, err)) return check(false, name + ": " + err);
    std::vector<uint8_t> card((size_t) bytes);
    dense::gather_spans(gg.tensor_data(*t), spans, card.data());
    const float* full = (const float*) gg.tensor_data(*t);
    const float* cut = (const float*) card.data();
    const int64_t last = s.shard_rows() - 1;
    bool first = cut[0] == full[global_at(s.runs, 0) * s.cols];
    bool end = cut[last * s.cols] == full[global_at(s.runs, last) * s.cols];
    const int64_t mid = s.shard_rows() / 3;   // a row past the first run, where the plan's packing changed hands
    bool middle = cut[mid * s.cols] == full[global_at(s.runs, mid) * s.cols];
    check(first && middle && end, name + " (card " + std::to_string(rank) + "): its rows are the ones it owns");
}

void plan_arithmetic() {
    std::string err;
    dense::Geometry g;   // the 27B's shape, by hand, so this half runs with no pack
    g.n_layers = 2;
    g.n_embd = 5120;
    g.n_ff = 17408;
    g.n_vocab = 248320;
    g.n_head = 24;
    g.n_kv = 4;
    g.head_dim = 256;
    g.gdn_state = 128;
    g.gdn_k_heads = 16;
    g.gdn_v_heads = 48;
    g.gdn_conv = 4;
    g.kinds = {dense::Mixer::Gdn, dense::Mixer::Attention};
    check(tp_geometry(g, 2, err), "the 27B's heads and widths share out between two cards");
    dense::Geometry odd = g;
    odd.n_kv = 3;
    err.clear();
    check(!tp_geometry(odd, 2, err) && !err.empty(), "three key/value heads refuse the pair");
    odd = g;
    odd.gdn_k_heads = 15;
    err.clear();
    check(!tp_geometry(odd, 2, err), "fifteen delta-net key heads refuse the pair");

    const int64_t AQ = g.n_head * g.head_dim;
    MatrixShard q[2];
    for (int rank = 0; rank < 2; ++rank) {
        err.clear();
        if (!tp_shard(g, "blk.3.attn_q.weight", 2 * AQ, g.n_embd, 2, rank, q[rank], err)) check(false, "attn_q plans: " + err);
    }
    check(partitions(q[0].runs, q[1].runs, 2 * AQ), "the query heads and their gates divide between the cards");
    check(q[0].run_values() == AQ && q[1].run_values() == AQ, "each card reads half the query projection's rows");

    MatrixShard qkv[2], out[2];
    for (int rank = 0; rank < 2; ++rank) {
        err.clear();
        if (!tp_shard(g, "blk.0.attn_qkv.weight", g.conv_channels(), g.n_embd, 2, rank, qkv[rank], err))
            check(false, "attn_qkv plans: " + err);
        err.clear();
        if (!tp_shard(g, "blk.0.ssm_out.weight", g.n_embd, g.value_width(), 2, rank, out[rank], err))
            check(false, "ssm_out plans: " + err);
    }
    check(partitions(qkv[0].runs, qkv[1].runs, g.conv_channels()), "the delta-net's channels follow its key-head groups");
    check(partitions(out[0].runs, out[1].runs, g.value_width()), "and its output's value columns with them");
    check(qkv[0].shard_rows() == g.conv_channels() / 2, "each card's slice is a delta-net layer of half the channels");
    check(out[0].dim == ShardDim::Cols && out[1].shard_cols() == g.value_width() / 2, "its output splits as its input columns");

    MatrixShard s;
    err.clear();
    check(!tp_shard(g, "blk.0.something_new.weight", 512, g.n_embd, 2, 0, s, err) && err.find("no rule") != std::string::npos,
          "a matrix the rules do not name is refused");
    err.clear();
    check(tp_shard(g, "blk.0.attn_norm.weight", g.n_embd, 1, 2, 1, s, err) && s.whole(), "a norm stays whole on both cards");
    err.clear();
    check(tp_shard(g, "blk.0.attn_qkv.scale", 1, 1, 2, 1, s, err) && s.whole(), "a matrix's tensor scale stays whole");
    err.clear();
    check(tp_shard(g, "dflash.blk.0.attn_conv.base", 2, g.n_embd, 2, 1, s, err) && s.whole(), "the drafter stays one card's");
}

int pack_checks(const std::string& path) {
    core::ModelFile file;   // a .basalt pack keeps its GGUF inside: the reader registers it as the tensor view
    std::string err;
    if (!file.open(path, err) || !file.load_meta(err) || !file.register_gguf_view(err)) {
        std::printf("# %s: %s\n", path.c_str(), err.c_str());
        return 1;
    }
    GgufFile gg(path);
    const dense::Geometry g = geometry_meta(gg, err);   // the shape, without the kernels' say so
    if (!err.empty()) {
        std::printf("# %s: %s\n", path.c_str(), err.c_str());
        return 1;
    }
    std::printf("# %s: %d layers, %zu tensors, a pair of cards\n", path.c_str(), g.n_layers, gg.tensors().size());
    int split = 0, shared = 0, alone = 0;
    bool rules = true, halves = true, tiling = true;
    std::string why;
    // what a card's load would place in VRAM: its shard of a split tensor, the whole of a shared one, the drafter's alone
    // on the card that runs it (the token embedding is read from page-locked RAM, so it is on neither card)
    uint64_t card_bytes[2] = {0, 0}, drafter = 0, mtp = 0, split_bytes = 0, shared_bytes = 0;
    for (const TensorInfo& t : gg.tensors()) {
        int64_t rows = 0, cols = 0;
        shape_of(gg, t.name, rows, cols);
        MatrixShard s[2];
        for (int rank = 0; rank < 2 && rules; ++rank) {
            err.clear();
            rules = tp_shard(g, t.name, rows, cols, 2, rank, s[rank], err);
            if (!rules) why = err;
        }
        if (!rules) break;
        if (tp_one_card(t.name)) {
            ++alone;
            std::string e;
            if (!s[0].whole() || dense::shard_bytes(t.type, s[0], e) != (int64_t) tensor_payload_bytes(t)) {
                why = t.name + ": one card's tensor does not stay whole";
                halves = false;
                break;
            }
            if (t.name != "token_embd.weight" && !t.name.starts_with("mtp.")) card_bytes[0] += tensor_payload_bytes(t);
            if (t.name.starts_with("dflash.")) drafter += tensor_payload_bytes(t);
            else if (t.name.starts_with("mtp.")) mtp += tensor_payload_bytes(t);   // card 0's instead, with --spec
            continue;
        }
        if (s[0].whole()) {
            ++shared;
            card_bytes[0] += tensor_payload_bytes(t);
            card_bytes[1] += tensor_payload_bytes(t);
            shared_bytes += tensor_payload_bytes(t);
            continue;
        }
        ++split;
        if (s[0].shard_rows() * s[0].shard_cols() * 2 != rows * cols) {
            why = t.name + ": " + std::to_string(s[0].shard_rows()) + " x " + std::to_string(s[0].shard_cols()) + " of " +
                  std::to_string(rows) + " x " + std::to_string(cols);
            halves = false;
            break;
        }
        if (!partitions(s[0].runs, s[1].runs, s[0].dim == ShardDim::Rows ? rows : cols)) {
            why = t.name + ": the cards do not share its cut dimension out";
            halves = false;
            break;
        }
        err.clear();
        if (!tiles(t, s, err)) {
            tiling = false;
            why = err;
            break;
        }
        std::string e2;
        card_bytes[0] += (uint64_t) dense::shard_bytes(t.type, s[0], e2);
        card_bytes[1] += (uint64_t) dense::shard_bytes(t.type, s[1], e2);
        split_bytes += tensor_payload_bytes(t);
    }
    check(rules, rules ? "every tensor of the pack has a shard rule" : "a tensor has no shard rule: " + why);
    check(halves, halves ? "every shard is half its matrix, runs and all" : "a shard is not half its matrix: " + why);
    check(tiling, tiling ? "the two cards' spans cut every tensor of the pack in two" : "a tensor's spans do not tile it: " + why);
    std::printf("# %d tensors split, %d shared whole, %d one card's\n", split, shared, alone);
    // the load's own per-card byte totals, cross-checked against the pack's sums: the second card holds the tensors both
    // hold plus half of every split one.  Not the rig's fit (plan_split prices that on the cards' free memory) - this is
    // that the loader's arithmetic adds up to the same bytes the shard rules promise.
    check(card_bytes[1] == shared_bytes + split_bytes / 2,
          "card 1's load is the shared tensors plus half the split ones: " + std::to_string(card_bytes[1] >> 20) + " MiB");
    std::printf("# per card: %.2f GiB on card 0 (%.2f GiB of it the drafter alone; the pack's drafter layer would be %.2f GiB with --spec), "
                "%.2f GiB on card 1, of %.2f GiB of weights (the token embedding's %.2f GiB stays in page-locked RAM)\n",
                (double) card_bytes[0] / 1073741824.0, (double) drafter / 1073741824.0, (double) mtp / 1073741824.0,
                (double) card_bytes[1] / 1073741824.0,
                (double) (split_bytes + shared_bytes) / 1073741824.0,
                (double) (gg.find("token_embd.weight") ? tensor_payload_bytes(*gg.find("token_embd.weight")) : 0) / 1073741824.0);

    planes_match(gg, g, "blk.0.attn_qkv.weight", 1, 97, 131);
    planes_match(gg, g, "blk.0.ffn_down.weight", 1, 7, 257);
    planes_match(gg, g, "blk.3.attn_q.weight", 1, 101, 131);
    planes_match(gg, g, "blk.0.ssm_out.weight", 1, 11, 193);
    floats_match(gg, g, "output.scale", 1);
    floats_match(gg, g, "blk.0.ssm_conv1d.weight", 1);
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const char* env = std::getenv("BASALT_QWEN35_PACK");
    const std::string pack = argc > 1 ? argv[1] : (env ? std::string(env) : std::string());
    plan_arithmetic();
    if (pack.empty()) {
        std::printf("# no pack: the tensor table and the byte-level checks are skipped\n");
        return failures == 0 ? 77 : 1;
    }
    return pack_checks(pack);
}
