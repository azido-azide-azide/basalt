// src/dense/tp.cpp - the tensor-parallel plan: card geometries, weight slice runs, and the byte fragments of a pack
// tensor each slice is.  Pure CPU: a load uses payload_fragments to pread and repack only the bytes its card holds.
#include "basalt/dense/tp.hpp"

#include "basalt/kernels/nvfp4_planes.hpp"

#include <algorithm>
#include <cstring>

namespace basalt::dense::tp {

std::string check(const Geometry& g, int tp) {
    if (tp != 2) return "tensor parallelism supports 2 cards for now";
    const auto half = [](int n) { return n / 2; };
    if (g.n_head % tp) return "query heads must divide the cards";
    if (g.n_kv % tp) return "key-value heads must divide the cards";
    if (half(g.n_head) % half(g.n_kv)) return "the per-card head grouping would not be whole";
    if (half(g.n_head) / half(g.n_kv) > 8) return "the per-card head grouping exceeds 8";
    if (g.gdn_k_heads % tp) return "GDN key heads must divide the cards";
    if (g.gdn_v_heads % tp) return "GDN value heads must divide the cards";
    if (half(g.gdn_v_heads) % half(g.gdn_k_heads)) return "the per-card GDN value grouping would not be whole";
    if (g.n_ff % tp || half(g.n_ff) % 256) return "the FFN's per-card units must be whole 256-value blocks";
    if ((int64_t) half(g.n_kv) * g.head_dim % 128) return "the per-card k/v rows must be whole NVFP4 row tiles";
    return {};
}

RankPlan rank_plan(const Geometry& g, int tp, int rank, std::string& err) {
    RankPlan p;
    if (err = check(g, tp); !err.empty()) return p;
    if (rank < 0 || rank >= tp) {
        err = "card rank " + std::to_string(rank) + " is not in a " + std::to_string(tp) + "-card group";
        return p;
    }
    p.tp = tp;
    p.rank = rank;
    p.geom = g;
    p.geom.n_head /= tp;
    p.geom.n_kv /= tp;
    p.geom.n_ff /= tp;
    p.geom.gdn_k_heads /= tp;
    p.geom.gdn_v_heads /= tp;
    p.gdn = core::GdnHeadSlice{g.gdn_k_heads, g.gdn_v_heads, g.gdn_state, rank * (g.gdn_k_heads / tp), g.gdn_k_heads / tp};
    return p;
}

std::vector<core::GdnRun> RankPlan::qkv_rows() const { return core::gdn_qkv_runs(gdn); }
std::vector<core::GdnRun> RankPlan::v_rows(int64_t per) const { return core::gdn_vhead_runs(gdn, per); }
std::vector<core::GdnRun> RankPlan::out_cols() const { return core::gdn_vhead_runs(gdn, gdn.s); }
std::vector<core::GdnRun> RankPlan::q_rows() const {
    return {{(int64_t) rank * aq() * 2, aq() * 2}};   // [query | gate] pairs, in head order
}
std::vector<core::GdnRun> RankPlan::kv_rows() const { return {{(int64_t) rank * kv(), kv()}}; }
std::vector<core::GdnRun> RankPlan::o_cols() const { return {{(int64_t) rank * aq(), aq()}}; }
std::vector<core::GdnRun> RankPlan::ffn_rows() const { return {{(int64_t) rank * geom.n_ff, geom.n_ff}}; }
std::vector<core::GdnRun> RankPlan::ffn_cols() const { return {{(int64_t) rank * geom.n_ff, geom.n_ff}}; }

namespace {
bool in_runs(int64_t axis, const std::vector<core::GdnRun>& runs) {
    for (const core::GdnRun& r : runs)
        if (r.start < 0 || r.len <= 0 || r.start + r.len > axis) return false;
    return !runs.empty();
}
int64_t runs_len(const std::vector<core::GdnRun>& runs) {
    int64_t n = 0;
    for (const core::GdnRun& r : runs) n += r.len;
    return n;
}
}  // namespace

std::vector<Frag> payload_fragments(uint32_t type, int64_t rows, int64_t cols, const std::vector<core::GdnRun>& runs,
                                    bool col_axis) {
    const int64_t axis = col_axis ? cols : rows;
    if (rows <= 0 || cols <= 0 || !in_runs(axis, runs)) return {};
    std::vector<Frag> f;
    int64_t at = 0;   // bytes written so far (fragments come back in ascending dst order)

    if (!col_axis) {
        switch (type) {
            case kF32:
            case kBf16: {
                const uint64_t rb = (uint64_t) cols * (type == kF32 ? 4 : 2);
                for (const core::GdnRun& r : runs) f.push_back({(uint64_t) r.start * rb, at, (uint64_t) r.len * rb}), at += r.len * rb;
                break;
            }
            case kQ8_0: {
                if (cols % 32) return {};
                const int64_t rb = cols / 32 * 34;
                for (const core::GdnRun& r : runs) f.push_back({(uint64_t) r.start * rb, at, (uint64_t) r.len * rb}), at += r.len * rb;
                break;
            }
            case kNvfp4Planes: {
                if (!kernels::nvfp4p::fits(rows, cols)) return {};
                for (const core::GdnRun& r : runs)
                    if (r.start % 128 || r.len % 128) return {};   // whole 128-row scale tiles
                const int64_t G = cols / 64;
                for (const core::GdnRun& r : runs) f.push_back({(uint64_t) r.start * cols / 2, at, (uint64_t) r.len * cols / 2}), at += r.len * cols / 2;
                const uint64_t so = (uint64_t) rows * cols / 2;
                at = so;
                for (const core::GdnRun& r : runs) {
                    const uint64_t len = (uint64_t) (r.len / 128) * G * 512;
                    f.push_back({so + (uint64_t) (r.start / 128) * G * 512, at, len});
                    at += len;
                }
                break;
            }
            default:
                return {};
        }
        return f;
    }

    // column (K) axis: the slice's K
    const int64_t K2 = runs_len(runs);
    switch (type) {
        case kF32:
        case kBf16: {
            const uint64_t es = type == kF32 ? 4 : 2;
            for (int64_t r = 0; r < rows; ++r) {
                for (const core::GdnRun& k : runs) {
                    f.push_back({((uint64_t) r * cols + k.start) * es, at, (uint64_t) k.len * es});
                    at += k.len * es;
                }
            }
            break;
        }
        case kQ8_0: {
            if (cols % 32 || K2 % 32) return {};
            for (const core::GdnRun& k : runs)
                if (k.start % 32 || k.len % 32) return {};
            const uint64_t rb = (uint64_t) cols / 32 * 34, rb2 = (uint64_t) K2 / 32 * 34;
            for (int64_t r = 0; r < rows; ++r) {
                int64_t db = 0;
                for (const core::GdnRun& k : runs) {
                    f.push_back({(uint64_t) r * rb + (uint64_t) (k.start / 32) * 34, r * rb2 + db, (uint64_t) (k.len / 32) * 34});
                    db += (k.len / 32) * 34;
                }
                at = (r + 1) * rb2;
            }
            break;
        }
        case kNvfp4Planes: {
            if (!nvfp4_rows_cols(rows, cols)) return {};
            for (const core::GdnRun& k : runs)
                if (k.start % 64 || k.len % 64) return {};   // whole 4-group scale tiles
            const int64_t G = cols / 64, G2 = K2 / 64;
            for (int64_t r = 0; r < rows; ++r) {
                int64_t dc = 0;
                for (const core::GdnRun& k : runs) {
                    f.push_back({(uint64_t) r * cols / 2 + k.start / 2, (uint64_t) r * K2 / 2 + dc / 2, (uint64_t) k.len / 2});
                    dc += k.len;
                }
            }
            const uint64_t so = (uint64_t) rows * cols / 2, so2 = (uint64_t) rows * K2 / 2;
            for (int64_t b = 0; b < rows / 128; ++b) {
                int64_t dt = 0;
                for (const core::GdnRun& k : runs) {
                    f.push_back({so + ((uint64_t) b * G + k.start / 64) * 512, so2 + ((uint64_t) b * G2 + dt / 64) * 512,
                                 (uint64_t) (k.len / 64) * 512});
                    dt += k.len;
                }
            }
            at = so2 + (uint64_t) rows * K2 / 64 * 512;
            break;
        }
        default:
            return {};
    }
    std::sort(f.begin(), f.end(), [](const Frag& a, const Frag& b) { return a.dst < b.dst; });
    return f;
}

uint64_t shard_bytes(const std::vector<Frag>& frags) {
    uint64_t n = 0;
    for (const Frag& f : frags) n += f.len;
    return n;
}

void gather_payload(const void* full, const std::vector<Frag>& frags, void* slice) {
    const uint8_t* s = (const uint8_t*) full;
    uint8_t* d = (uint8_t*) slice;
    for (const Frag& f : frags) std::memcpy(d + f.dst, s + f.src, f.len);
}

}  // namespace basalt::dense::tp
