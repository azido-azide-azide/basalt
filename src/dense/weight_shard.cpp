// src/dense/weight_shard.cpp - see include/basalt/dense/weight_shard.hpp.
#include "basalt/dense/weight_shard.hpp"

#include "basalt/kernels/nvfp4_planes.hpp"

#include <cstring>

namespace basalt::dense {

namespace {

constexpr uint32_t kTypeF32 = 0, kTypeQ8_0 = 8, kTypeBf16 = 30, kTypeF8 = 44, kTypeNvfp4Planes = 45;

/// A type's block along K: `values` values in `bytes` bytes (F32, BF16 and F8_E4M3 carry one value per block).
bool block_of(uint32_t type, int64_t& values, int64_t& bytes) {
    switch (type) {
    case kTypeF32:
        values = 1;
        bytes = 4;
        return true;
    case kTypeBf16:
        values = 1;
        bytes = 2;
        return true;
    case kTypeF8:
        values = 1;
        bytes = 1;
        return true;
    case kTypeQ8_0:
        values = 32;
        bytes = 34;
        return true;
    default:
        return false;
    }
}

std::string shape(int64_t rows, int64_t cols) {
    return std::to_string((long long) rows) + " x " + std::to_string((long long) cols);
}

std::string cut_at(int64_t at) {
    return "not at element " + std::to_string((long long) at);
}

int64_t row_bytes(int64_t cols, int64_t values, int64_t bytes) { return (cols / values) * bytes; }

}  // namespace

int64_t MatrixShard::run_values() const {
    int64_t n = 0;
    for (const ShardRun& r : runs) n += r.count;
    return n;
}

int64_t MatrixShard::shard_rows() const { return dim == ShardDim::Rows ? run_values() : rows; }

int64_t MatrixShard::shard_cols() const { return dim == ShardDim::Cols ? run_values() : cols; }

namespace {

/// The card's rows of a matrix whose K is whole, as spans of its own copy (its rows are as wide as the whole one's).
void row_spans(const std::vector<ShardRun>& runs, int64_t rb_full, std::vector<ByteSpan>& out) {
    int64_t at = 0;
    for (const ShardRun& r : runs) {
        out.push_back({r.first * rb_full, at, r.count * rb_full});
        at += r.count * rb_full;
    }
}

/// The card's columns of every row of a matrix whose rows are whole, packed into its narrower rows.
void col_spans(int64_t rows, int64_t rb_full, int64_t rb_card, const std::vector<ShardRun>& runs, int64_t values,
               int64_t bytes, std::vector<ByteSpan>& out) {
    for (int64_t i = 0; i < rows; ++i) {
        int64_t at = 0;
        for (const ShardRun& r : runs) {
            const int64_t n = (r.count / values) * bytes;
            out.push_back({i * rb_full + (r.first / values) * bytes, i * rb_card + at, n});
            at += n;
        }
    }
}

/// NVFP4 planes: the codes [rows][cols / 2], then the scale - one UE4M3 per 16 values - in 512-byte tiles of 128 rows
/// x 4 groups, row-block major.  Whole 128-row blocks and whole 64-value groups keep both parts in one piece.
bool planes_spans(const MatrixShard& s, std::vector<ByteSpan>& out, std::string& err) {
    const int64_t r = s.rows, k = s.cols, rc = s.shard_rows(), kc = s.shard_cols();
    if (!kernels::nvfp4p::fits(r, k) || !kernels::nvfp4p::fits(rc, kc)) {
        err = "NVFP4 planes of " + shape(r, k) + " and of the card's " + shape(rc, kc) + " (planes need rows in 128s and 64-value groups)";
        return false;
    }
    const int64_t scales_full = (int64_t) kernels::nvfp4p::scale_offset(r, k);
    const int64_t scales_card = (int64_t) kernels::nvfp4p::scale_offset(rc, kc);
    const int64_t wide_full = k / 64, wide_card = kc / 64;
    if (s.dim == ShardDim::Rows) {
        int64_t codes = 0, scales = 0;
        for (const ShardRun& run : s.runs) {
            if (run.first % 128 || run.count % 128) {
                err = std::string("an NVFP4 planes matrix takes its rows in whole 128-row blocks, ") + cut_at(run.first);
                return false;
            }
            const int64_t cn = run.count * k / 2;
            out.push_back({run.first * k / 2, codes, cn});
            codes += cn;
            const int64_t sn = (run.count / 128) * wide_full * 512;
            out.push_back({scales_full + (run.first / 128) * wide_full * 512, scales_card + scales, sn});
            scales += sn;
        }
        return true;
    }
    for (const ShardRun& run : s.runs)
        if (run.first % 64 || run.count % 64) {
            err = std::string("an NVFP4 planes matrix takes its K in whole 64-value groups, ") + cut_at(run.first);
            return false;
        }
    for (int64_t i = 0; i < r; ++i) {
        int64_t at = 0;
        for (const ShardRun& run : s.runs) {
            out.push_back({i * (k / 2) + run.first / 2, i * (kc / 2) + at / 2, run.count / 2});
            at += run.count;
        }
    }
    for (int64_t block = 0; block < r / 128; ++block) {
        int64_t at = 0;
        for (const ShardRun& run : s.runs) {
            out.push_back({scales_full + (block * wide_full + run.first / 64) * 512,
                           scales_card + (block * wide_card + at / 64) * 512, (run.count / 64) * 512});
            at += run.count;
        }
    }
    return true;
}

}  // namespace

bool shard_spans(uint32_t type, const MatrixShard& s, std::vector<ByteSpan>& out, std::string& err) {
    out.clear();
    err.clear();
    if (s.rows <= 0 || s.cols <= 0) {
        err = "a matrix shard of " + shape(s.rows, s.cols);
        return false;
    }
    if (s.dim == ShardDim::None) {
        const int64_t n = shard_bytes(type, s, err);
        if (n <= 0) return false;
        out.push_back({0, 0, n});
        return true;
    }
    if (s.runs.empty()) {
        err = "a shard that cuts a matrix needs the runs it holds";
        return false;
    }
    if (type == kTypeNvfp4Planes) return planes_spans(s, out, err);
    int64_t values = 0, bytes = 0;
    if (!block_of(type, values, bytes)) {
        err = "a matrix shard of ggml type " + std::to_string(type);
        return false;
    }
    const int64_t kc = s.shard_cols();
    if (s.cols % values || kc % values) {
        err = "a matrix of " + std::to_string((long long) s.cols) + " values needs K in whole " + std::to_string(values) +
              "-value blocks, the card's " + std::to_string((long long) kc);
        return false;
    }
    const int64_t rb_full = row_bytes(s.cols, values, bytes), rb_card = row_bytes(kc, values, bytes);
    if (s.dim == ShardDim::Rows) {
        row_spans(s.runs, rb_full, out);
        return true;
    }
    for (const ShardRun& r : s.runs)
        if (r.first % values || r.count % values) {
            err = "a matrix of ggml type " + std::to_string(type) + " takes its K in whole " + std::to_string(values) +
                  "-value blocks, " + cut_at(r.first);
            return false;
        }
    col_spans(s.rows, rb_full, rb_card, s.runs, values, bytes, out);
    return true;
}

int64_t shard_bytes(uint32_t type, const MatrixShard& s, std::string& err) {
    const int64_t r = s.shard_rows(), c = s.shard_cols();
    err.clear();
    if (r <= 0 || c <= 0) {
        err = "a matrix shard of " + shape(r, c);
        return 0;
    }
    if (type == kTypeNvfp4Planes) {
        if (!kernels::nvfp4p::fits(r, c)) {
            err = "NVFP4 planes of " + shape(r, c) + " (planes need rows in 128s and 64-value groups)";
            return 0;
        }
        return (int64_t) kernels::nvfp4p::bytes(r, c);
    }
    int64_t values = 0, bytes = 0;
    if (!block_of(type, values, bytes)) {
        err = "a matrix shard of ggml type " + std::to_string(type);
        return 0;
    }
    if (c % values) {
        err = "a matrix of " + std::to_string((long long) c) + " values needs K in whole " + std::to_string(values) + "-value blocks";
        return 0;
    }
    return r * row_bytes(c, values, bytes);
}

void gather_spans(const uint8_t* full, const std::vector<ByteSpan>& spans, uint8_t* out) {
    for (const ByteSpan& s : spans) std::memcpy(out + s.dst, full + s.src, (size_t) s.n);
}

std::vector<ShardRun> unit_runs(int64_t n_units, int64_t unit, int width, int rank) {
    if (n_units <= 0 || unit <= 0 || width <= 0 || rank < 0 || rank >= width || n_units % width) return {};
    const int64_t per = (n_units / width) * unit;
    return {{rank * per, per}};
}

void append_runs(std::vector<ShardRun>& to, const std::vector<ShardRun>& runs, int64_t offset) {
    for (const ShardRun& r : runs) to.push_back({r.first + offset, r.count});
}

}  // namespace basalt::dense
