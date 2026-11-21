#pragma once
// include/basalt/core/gdn_head_split.hpp - the GDN mixer split by key-head group: the rows, columns and state a card
// holds when it owns key heads [k0, k0 + nk) of a GDN layer.
//
// Value head v reads key head v % h_k, so the card's value heads are the runs [j * h_k + k0, j * h_k + k0 + nk) for
// j < h_v / h_k, packed in that order.  Packed so, the slice is a GDN layer of nk key heads and nk * h_v / h_k value
// heads: the qkv/z projections are row slices, dec_gdn_rec runs unchanged with the sliced head counts, and the out
// projection over the slice's columns gives a partial sum of the layer's output.  Quantized out-projection columns
// slice on whole blocks only (gdn_slice_aligned).
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__CUDACC__)
#define GDN_HEAD_SPLIT_HD __host__ __device__
#else
#define GDN_HEAD_SPLIT_HD
#endif

namespace basalt::core {

struct GdnHeadSlice {
    int h_k = 0, h_v = 0, s = 0;   // the layer's key heads, value heads and head size
    int k0 = 0, nk = 0;            // the key heads this card owns
    int rep() const { return h_v / h_k; }
    int nv() const { return nk * rep(); }
    int64_t c() const { return (int64_t) (2 * nk + nv()) * s; }   // packed qkv / conv channels
    int64_t zv() const { return (int64_t) nv() * s; }             // packed value width (z rows, out columns)
};

struct GdnRun {
    int64_t start, len;
};

// channel runs of the full qkv axis [q | k | v], in packed order
inline std::vector<GdnRun> gdn_qkv_runs(const GdnHeadSlice& g) {
    const int64_t s = g.s;
    std::vector<GdnRun> r{{g.k0 * s, g.nk * s}, {(g.h_k + g.k0) * s, g.nk * s}};
    for (int j = 0; j < g.rep(); ++j) r.push_back({2 * g.h_k * s + (int64_t) (j * g.h_k + g.k0) * s, g.nk * s});
    return r;
}

// runs over the value-head axis, `per` elements per head (1: gates, dt, ssm_a; s: z rows, out columns, state rows)
inline std::vector<GdnRun> gdn_vhead_runs(const GdnHeadSlice& g, int64_t per) {
    std::vector<GdnRun> r;
    for (int j = 0; j < g.rep(); ++j) r.push_back({(int64_t) (j * g.h_k + g.k0) * per, g.nk * per});
    return r;
}

/// THE SAME CUT READ AS A HEAD MAP: the model value head behind one of the card's PACKED local heads.  The
/// packed order is `gdn_vhead_runs`' own - run j holds model heads [j * h_k + k0, j * h_k + k0 + nk) - so
/// packed head p = j * nk + i is model head j * h_k + k0 + i.
///
/// Everything that couples a card's sliced activations to a FULL-width object (a replicated per-head
/// parameter row, a replicated output projection's columns) is addressed through THIS, never through
/// `rank * nv + p`: a key-head-group cut is not a contiguous one, because value head v reads key head
/// v % h_k, so the card's heads stride the model's head axis in steps of h_k.  With h_v / h_k == 1 the two
/// agree and the distinction vanishes; with 48 value heads over 16 key heads it does not.
GDN_HEAD_SPLIT_HD inline int64_t gdn_local_head(const GdnHeadSlice& g, int64_t p) {
    return (p / g.nk) * g.h_k + g.k0 + p % g.nk;
}

/// the column of the FULL value row (h_v * s wide) where the card's packed local head p belongs
GDN_HEAD_SPLIT_HD inline int64_t gdn_local_col(const GdnHeadSlice& g, int64_t p) {
    return gdn_local_head(g, p) * g.s;
}

/// that placement as the three strides a kernel walking the full row needs: the card's rows arrive in runs
/// of `src_group` = nk * s elements and land inside runs of `dst_group` = h_k * s, at `base` = k0 * s.
GDN_HEAD_SPLIT_HD inline void gdn_place_groups(const GdnHeadSlice& g, int64_t& src_group, int64_t& dst_group,
                                               int64_t& base) {
    src_group = (int64_t) g.nk * g.s;
    dst_group = (int64_t) g.h_k * g.s;
    base = (int64_t) g.k0 * g.s;
}

/// the source column a grouped placement draws for full-row column `c`, or -1 where the card holds no value
/// (the placement's index law in ONE place: `tp_place_*_grouped` runs it on the device, a host test runs it
/// against `gdn_local_col` - the pair's wrong-output defect was exactly these two views disagreeing)
GDN_HEAD_SPLIT_HD inline int64_t gdn_grouped_src_col(int64_t c, int64_t src_group, int64_t dst_group,
                                                     int64_t base) {
    const int64_t grp = c / dst_group, rem = c - grp * dst_group;
    if (rem < base || rem >= base + src_group) return -1;
    return grp * src_group + (rem - base);
}

inline int64_t gdn_runs_len(const std::vector<GdnRun>& runs) {
    int64_t n = 0;
    for (const GdnRun& r : runs) n += r.len;
    return n;
}

// true when every value-head run of the out projection starts and ends on a `block_cols` boundary
inline bool gdn_slice_aligned(const GdnHeadSlice& g, int64_t block_cols) {
    for (const GdnRun& r : gdn_vhead_runs(g, g.s))
        if (r.start % block_cols != 0 || r.len % block_cols != 0) return false;
    return true;
}

// rows `runs` of a row-major matrix of `row_bytes` per row, packed into dst
inline void gdn_gather_rows(const void* src, size_t row_bytes, const std::vector<GdnRun>& runs, void* dst) {
    const auto* s = static_cast<const uint8_t*>(src);
    auto* d = static_cast<uint8_t*>(dst);
    for (const GdnRun& r : runs) {
        std::memcpy(d, s + (size_t) r.start * row_bytes, (size_t) r.len * row_bytes);
        d += (size_t) r.len * row_bytes;
    }
}

// columns `runs` of each of n_rows rows of a blocked row-major matrix (block_cols values in block_bytes), packed into
// dst; every run lies on whole blocks
inline void gdn_gather_cols(const void* src, int64_t n_rows, int64_t n_cols, int64_t block_cols, size_t block_bytes,
                            const std::vector<GdnRun>& runs, void* dst) {
    const auto* s = static_cast<const uint8_t*>(src);
    auto* d = static_cast<uint8_t*>(dst);
    const size_t row_bytes = (size_t) (n_cols / block_cols) * block_bytes;
    for (int64_t i = 0; i < n_rows; ++i)
        for (const GdnRun& r : runs) {
            const size_t n = (size_t) (r.len / block_cols) * block_bytes;
            std::memcpy(d, s + (size_t) i * row_bytes + (size_t) (r.start / block_cols) * block_bytes, n);
            d += n;
        }
}

// floats of one layer's sliced state: the recurrent state [s][nv][s], then the conv history [c][hist]
inline size_t gdn_slice_state_floats(const GdnHeadSlice& g, int hist) {
    return (size_t) g.s * g.nv() * g.s + (size_t) g.c() * hist;
}

// the full layer state ([s][h_v][s], then the conv history [C][hist]) to the slice's layout
inline void gdn_slice_state(const GdnHeadSlice& g, int hist, const float* full, float* out) {
    const std::vector<GdnRun> vr = gdn_vhead_runs(g, g.s);
    for (int i = 0; i < g.s; ++i)
        gdn_gather_rows(full + (size_t) i * g.h_v * g.s, sizeof(float), vr, out + (size_t) i * g.nv() * g.s);
    gdn_gather_rows(full + (size_t) g.s * g.h_v * g.s, sizeof(float) * hist, gdn_qkv_runs(g),
                    out + (size_t) g.s * g.nv() * g.s);
}

// the slice's state back into the full layer state (the inverse of gdn_slice_state on the slice's heads)
inline void gdn_unslice_state(const GdnHeadSlice& g, int hist, const float* sliced, float* full) {
    const std::vector<GdnRun> vr = gdn_vhead_runs(g, g.s);
    for (int i = 0; i < g.s; ++i) {
        const float* src = sliced + (size_t) i * g.nv() * g.s;
        for (const GdnRun& r : vr) {
            std::memcpy(full + (size_t) i * g.h_v * g.s + r.start, src, (size_t) r.len * sizeof(float));
            src += r.len;
        }
    }
    const float* src = sliced + (size_t) g.s * g.nv() * g.s;
    float* hfull = full + (size_t) g.s * g.h_v * g.s;
    for (const GdnRun& r : gdn_qkv_runs(g)) {
        std::memcpy(hfull + (size_t) r.start * hist, src, (size_t) r.len * hist * sizeof(float));
        src += (size_t) r.len * hist;
    }
}

}  // namespace basalt::core
