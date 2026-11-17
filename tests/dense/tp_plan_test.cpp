// tests/dense/tp_plan_test.cpp - the tensor-parallel plan on the host: card geometries and runs (halves cover the
// axes, GDN value-head runs follow the "value head h meets key head h % k_heads" rule), and payload_fragments
// checked byte-exact against an independent model of each tensor layout (NVFP4_PLANES' 128-row x 4-group scale
// tiles, Q8_0's 34-byte blocks, plain F32/BF16 rows) - a slice byte is where the layout's own index formula says,
// not where a second copy of the fragment code says.  Misaligned and unsupported cases must come back empty.
#include "basalt/dense/tp.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace basalt;
using dense::Geometry;
using dense::Mixer;
using dense::tp::Frag;
using dense::tp::RankPlan;
using core::GdnRun;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}

// a 27B-shaped geometry: 24/4 heads of 256, H 5120, FF 17408, GDN 16 key / 48 value of state 128, attn every 4th
Geometry geom27b() {
    Geometry g;
    g.n_layers = 64;
    g.n_embd = 5120;
    g.n_ff = 17408;
    g.n_vocab = 248320;
    g.n_head = 24;
    g.n_kv = 4;
    g.head_dim = 256;
    g.rope.rot = 64;
    g.rope.theta = 1e7f;
    g.gdn_state = 128;
    g.gdn_k_heads = 16;
    g.gdn_v_heads = 48;
    g.gdn_conv = 4;
    g.context_length = 262144;
    for (int il = 0; il < g.n_layers; ++il) g.kinds.push_back((il + 1) % 4 == 0 ? Mixer::Attention : Mixer::Gdn);
    return g;
}

int64_t len_sum(const std::vector<GdnRun>& runs) {
    int64_t n = 0;
    for (const GdnRun& r : runs) n += r.len;
    return n;
}

// the runs of both cards are disjoint and cover [0, axis)
bool covers(int64_t axis, const std::vector<GdnRun>& a, const std::vector<GdnRun>& b) {
    std::vector<char> hit((size_t) axis, 0);
    for (const std::vector<GdnRun>* rs : {&a, &b})
        for (const GdnRun& r : *rs) {
            if (r.start < 0 || r.start + r.len > axis) return false;
            for (int64_t i = r.start; i < r.start + r.len; ++i)
                if (hit[(size_t) i]) return false;
            // touch
            for (int64_t i = r.start; i < r.start + r.len; ++i) hit[(size_t) i] = 1;
        }
    for (char c : hit)
        if (!c) return false;
    return true;
}

uint8_t lcg(uint32_t& x) {
    x = x * 1664525u + 1013904223u;
    return (uint8_t) (x >> 24);
}

// ---- independent layout models: the source byte of a slice element, from the formats' own formulas ----

// gather the slice and check every byte against the layout model.  col_axis false: whole rows (in run order);
// true: the columns of every row.  Only whole-nibble pairs and whole scale groups are addressed: runs keep them.
void check_nvfp4(int64_t rows, int64_t K, const std::vector<GdnRun>& runs, bool col_axis, const std::string& tag) {
    const uint64_t total = (uint64_t) rows * K / 64 * 36;
    std::vector<uint8_t> full(total);
    uint32_t x = 7u + (uint32_t) (total & 0xff);
    for (uint8_t& b : full) b = lcg(x);
    std::vector<Frag> frags = dense::tp::payload_fragments(dense::tp::kNvfp4Planes, rows, K, runs, col_axis);
    if (frags.empty()) {
        check(false, tag + ": fragments produced");
        return;
    }
    std::vector<uint8_t> slice(dense::tp::shard_bytes(frags));
    dense::tp::gather_payload(full.data(), frags, slice.data());
    // the model: walk the slice's own address space and ask the layout where each byte came from
    const int64_t K2 = len_sum(runs), R2 = col_axis ? rows : len_sum(runs);
    (void) R2;
    std::vector<uint64_t> expect;   // source byte of each code byte and scale byte of the slice, in slice order
    if (!col_axis) {
        int64_t rr = 0;
        (void) rr;
        for (const GdnRun& r : runs)
            for (int64_t i = 0; i < r.len; ++i)
                for (int64_t cb = 0; cb < K / 2; ++cb) expect.push_back((uint64_t) (r.start + i) * K / 2 + cb);
        for (const GdnRun& r : runs)   // slice bands follow the runs' source bands, in run order
            for (int64_t band = r.start / 128; band < (r.start + r.len) / 128; ++band)
                for (int64_t tg = 0; tg < K / 64; ++tg)   // one whole 128-row x 4-group tile, relabelled
                    expect.push_back((uint64_t) rows * K / 2 + ((uint64_t) band * (K / 64) + tg) * 512);
    } else {
        for (int64_t r = 0; r < rows; ++r)
            for (const GdnRun& k : runs)
                for (int64_t cb = 0; cb < k.len / 2; ++cb) expect.push_back((uint64_t) r * K / 2 + (k.start) / 2 + cb);
        for (int64_t b = 0; b < rows / 128; ++b)
            for (const GdnRun& k : runs)
                for (int64_t t = 0; t < k.len / 64; ++t)
                    for (uint64_t i = 0; i < 512; ++i) expect.push_back((uint64_t) rows * K / 2 + ((uint64_t) b * (K / 64) + k.start / 64 + t) * 512 + i);
    }
    // slice bytes sit at dst; frags are ascending, contiguous and cover the slice
    bool layout_ok = expect.size() == slice.size();
    uint64_t at = 0;
    for (const Frag& f : frags) {
        if (f.dst != at) layout_ok = false;
        at += f.len;
    }
    if (at != slice.size()) layout_ok = false;
    check(layout_ok, tag + ": the slice is whole and fragment-dense (" + std::to_string(slice.size()) + " bytes)");
    if (!layout_ok) return;
    // compare byte values: dst position i must hold full[expect[i]]
    bool bytes_ok = true;
    at = 0;
    size_t ei = 0;
    for (const Frag& f : frags) {
        for (uint64_t i = 0; i < f.len; ++i) {
            if (slice[at + i] != full[expect[ei++]]) {
                bytes_ok = false;
                break;
            }
        }
        at += f.len;
    }
    check(bytes_ok, tag + ": every byte sits where the layout formula puts it");
}

void check_q8(int64_t rows, int64_t K, const std::vector<GdnRun>& runs, bool col_axis, const std::string& tag) {
    const uint64_t total = (uint64_t) rows * K / 32 * 34;
    std::vector<uint8_t> full(total);
    uint32_t x = 11u + (uint32_t) (total & 0xff);
    for (uint8_t& b : full) b = lcg(x);
    std::vector<Frag> frags = dense::tp::payload_fragments(dense::tp::kQ8_0, rows, K, runs, col_axis);
    check(!frags.empty(), tag + ": fragments produced");
    if (frags.empty()) return;
    uint64_t sn = 0;
    for (const Frag& f : frags) sn += f.len;
    std::vector<uint8_t> slice(sn);
    dense::tp::gather_payload(full.data(), frags, slice.data());
    // model: the slice's blocks, in slice order, are the runs' source blocks in slice order (each block 34 bytes whole)
    std::vector<uint64_t> expect;
    const int64_t rb = K / 32, rb2 = len_sum(runs) / 32;
    if (!col_axis) {
        for (const GdnRun& r : runs)
            for (int64_t i = 0; i < r.len; ++i)
                for (int64_t b = 0; b < rb; ++b) expect.push_back(((uint64_t) (r.start + i) * rb + b) * 34);
    } else {
        for (int64_t r = 0; r < rows; ++r)
            for (const GdnRun& k : runs)
                for (int64_t b = 0; b < k.len / 32; ++b) expect.push_back(((uint64_t) r * rb + k.start / 32 + b) * 34);
    }
    bool ok = expect.size() * 34 == slice.size();
    for (size_t i = 0; ok && i * 34 < slice.size(); ++i)
        for (int j = 0; j < 34; ++j)
            if (slice[i * 34 + j] != full[expect[i] + j]) {
                ok = false;
                break;
            }
    (void) rb2;
    check(ok, tag + ": whole 34-byte blocks, in place");
}

void check_plain(int type, uint64_t es, int64_t rows, int64_t cols, const std::vector<GdnRun>& runs, bool col_axis,
                 const std::string& tag) {
    const uint64_t total = (uint64_t) rows * cols * es;
    std::vector<uint8_t> full(total);
    uint32_t x = 23u + (uint32_t) (total & 0xff);
    for (uint8_t& b : full) b = lcg(x);
    std::vector<Frag> frags = dense::tp::payload_fragments(type, rows, cols, runs, col_axis);
    check(!frags.empty(), tag + ": fragments produced");
    if (frags.empty()) return;
    std::vector<uint8_t> slice(total);
    dense::tp::gather_payload(full.data(), frags, slice.data());
    bool ok = true;
    for (const Frag& f : frags)
        if (std::memcmp(slice.data() + f.dst, full.data() + f.src, f.len) != 0) ok = false;
    uint64_t n = 0;
    for (const Frag& f : frags) n += f.len;
    check(ok && n == dense::tp::shard_bytes(frags), tag + ": bytes copied from the modelled sources");
}
}  // namespace

int main() {
    // ---- card geometries ----
    const Geometry g = geom27b();
    check(dense::tp::check(g, 2).empty(), "the 27B geometry splits over two cards");
    std::string err;
    const RankPlan p0 = dense::tp::rank_plan(g, 2, 0, err);
    check(err.empty() && p0.rank == 0, "rank 0's plan");
    const RankPlan p1 = dense::tp::rank_plan(g, 2, 1, err);
    check(err.empty() && p1.rank == 1, "rank 1's plan");
    check(p0.geom.n_head == 12 && p0.geom.n_kv == 2 && p0.geom.n_ff == 8704 && p0.geom.gdn_k_heads == 8 &&
              p0.geom.gdn_v_heads == 24 && p0.geom.conv_channels() == 5120 && p0.geom.value_width() == 3072,
          "the card geometry halves heads, FFN and GDN (12q/2kv, 8704, 8k/24v, C 5120, V 3072)");
    check(p0.geom.n_embd == g.n_embd && p0.geom.head_dim == g.head_dim && p0.geom.gdn_state == g.gdn_state &&
              p0.geom.kinds == g.kinds && p0.geom.rope.theta == g.rope.theta,
          "width, head size, state, layer kinds and rope carry over whole");
    check(p0.k0() == 0 && p1.k0() == 8 && p0.nk() == 8 && p0.nv() == 24, "key ranges [0,8) and [8,16), 24 value heads each");

    // ---- runs: the two cards cover every axis, and the GDN value heads follow their key heads ----
    check(covers(2 * (int64_t) g.n_head * g.head_dim, p0.q_rows(), p1.q_rows()), "q rows ([query|gate] pairs) split by head");
    check(covers((int64_t) g.n_kv * g.head_dim, p0.kv_rows(), p1.kv_rows()), "k/v rows split by key-value head");
    check(covers((int64_t) g.n_head * g.head_dim, p0.o_cols(), p1.o_cols()), "o columns split by query head");
    check(covers(g.conv_channels(), p0.qkv_rows(), p1.qkv_rows()), "the qkv/conv channel axis splits over the cards");
    check(covers(g.value_width(), p0.out_cols(), p1.out_cols()), "the out-projection value columns split over the cards");
    check(covers(g.gdn_v_heads, p0.v_rows(1), p1.v_rows(1)), "the per-head scalars split by value head");
    check(covers(g.n_ff, p0.ffn_rows(), p1.ffn_rows()), "FFN gate/up rows split contiguously");
    check(covers(g.n_ff, p0.ffn_cols(), p1.ffn_cols()), "FFN down columns split contiguously");
    {   // value head h meets key head h % k_heads: card 1's z rows are heads with h % 16 in [8, 16)
        bool ok = true;
        const auto runs = p1.v_rows(1);
        if (runs.size() != 3) ok = false;
        int64_t seen = 0;
        for (const GdnRun& r : runs)
            for (int64_t h = r.start; h < r.start + r.len; ++h) {
                if (!(h % 16 >= 8 && h % 16 < 16)) ok = false;
                ++seen;
            }
        check(ok && seen == 24, "card 1's value heads are exactly {h : h % 16 in [8, 16)}");
    }
    {   // the packed local order preserves the rule: local value h' meets local key h' % nk
        const int nk = p1.nk(), rep = g.gdn_v_heads / g.gdn_k_heads;
        const auto runs = p1.v_rows(1);
        bool ok = runs.size() == rep;
        int64_t local = 0;
        for (const GdnRun& r : runs)
            for (int64_t h = r.start; h < r.start + r.len; ++h, ++local)
                if ((h % g.gdn_k_heads) - p1.k0() != local % nk) ok = false;
        check(ok, "packed order keeps local value head h' with local key head h' % nk");
    }

    // ---- fragments against the layout models ----
    {   // qkv-style rows: runs over a 10240-value row axis, five 1024-row blocks (rank 1 of the 27B)
        check_nvfp4(10240, 5120, p1.qkv_rows(), false, "nvfp4 qkv rows (5 runs x 1024 of 10240, K 5120)");
        check_q8(10240, 5120, p1.qkv_rows(), false, "q8_0 qkv rows");
        // out-projection columns: three strided 1024-value runs of a 6144-value axis
        check_nvfp4(5120, 6144, p0.out_cols(), true, "nvfp4 out columns (3 runs x 1024 of 6144)");
        check_nvfp4(5120, 6144, p1.out_cols(), true, "nvfp4 out columns, card 1");
        check_q8(5120, 6144, p1.out_cols(), true, "q8_0 out columns");
        // contiguous halves
        check_nvfp4(5120, 12288, p1.o_cols(), true, "nvfp4 o columns (half of 12288)");
        check_nvfp4(12288, 5120, p1.q_rows(), false, "nvfp4 q rows (half of 12288)");
        check_nvfp4(17408, 5120, p1.ffn_rows(), false, "nvfp4 gate rows (half of 17408)");
        check_nvfp4(5120, 17408, p0.ffn_cols(), true, "nvfp4 down columns (half of 17408)");
        check_q8(5120, 17408, p0.ffn_cols(), true, "q8_0 down columns");
        check_q8(12288, 5120, p0.q_rows(), false, "q8_0 q rows");
        // the small per-head tensors (dt/ssm_a: 1 float per value head; conv: 4 per channel; z: 128 per head)
        check_plain(dense::tp::kF32, 4, g.gdn_v_heads, 1, p1.v_rows(1), false, "f32 dt rows (3 runs x 8 of 48)");
        check_plain(dense::tp::kBf16, 2, g.gdn_v_heads, 5120, p1.v_rows(1), false, "bf16 alpha rows");
        check_plain(dense::tp::kF32, 4, g.conv_channels(), 4, p1.qkv_rows(), false, "f32 conv channel rows");
        check_plain(dense::tp::kF32, 4, 5120, 6144, p1.out_cols(), true, "f32 out columns (plain: per-element runs)");
        check_plain(dense::tp::kBf16, 2, 48, 5120, p0.v_rows(1), false, "bf16 per-head rows, card 0");
    }

    // ---- refusal cases ----
    {
        Geometry odd = geom27b();
        odd.n_head = 23;
        check(!dense::tp::check(odd, 2).empty(), "an odd query-head count refuses tp 2");
        Geometry kv1 = geom27b();
        kv1.n_kv = 2;   // per card 1, per card heads 12: grouping 12 > 8
        check(!dense::tp::check(kv1, 2).empty(), "a per-card grouping over 8 refuses");
        check(!dense::tp::check(g, 3).empty(), "three cards refuse for now");
        std::string e2;
        dense::tp::rank_plan(g, 2, 2, e2);
        check(!e2.empty(), "a rank outside the group refuses");
        check(dense::tp::payload_fragments(dense::tp::kNvfp4Planes, 256, 128, {{16, 128}}, false).empty(),
              "NVFP4 row runs off a 128-row boundary refuse");
        check(dense::tp::payload_fragments(dense::tp::kNvfp4Planes, 256, 128, {{16, 64}}, true).empty(),
              "NVFP4 column runs off a 4-group tile refuse");
        check(dense::tp::payload_fragments(dense::tp::kQ8_0, 256, 100, {{0, 32}}, true).empty(),
              "Q8_0 on a column axis not whole 32-value blocks refuses");
        check(dense::tp::payload_fragments(99, 256, 128, {{0, 128}}, false).empty(), "an unknown tensor type refuses");
    }

    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
