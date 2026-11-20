// src/core/weights.cpp - the dense-weight loader.  See the header for the engine-vs-pack distinction.
#include "basalt/core/weights.hpp"
#include "basalt/core/direct_upload.hpp"
#include "basalt/core/load_progress.hpp"

#include "basalt/kernels/f16_bits.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cinttypes>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace basalt::core {
namespace {

constexpr uint64_t CHUNK = basalt::core::DirectUploader::kChunk;   // SOURCE bytes per staging round

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct IndexRow {
    std::string name;
    int file = 0;
    WeightKind kind = WeightKind::Verbatim;
    uint64_t src_off = 0, src_bytes = 0, dst_off = 0, dst_bytes = 0;
    int64_t ne0 = 0, ne1 = 0;
    int code_bits = 0, code_bias = 0, group_elems = 0, codebook = 0, has_offset = 0;
    uint64_t codes_bytes = 0, scales_bytes = 0, offset_bytes = 0;
    int scales_fp16 = 0;
    int act_kind = 0;
    bool raw16 = false;   ///< already 16-bit in the file (kinds 4/5)
    // Row slices (see WeightTable::RowSlices): the stored planes' byte spans, relative to `src_off`, in
    // (codes, scales, offsets) plane units with each span's length in FILE bytes.  Empty = the full tensor.
    // After a slice is applied the plane fields above and ne1/dst_bytes describe the STORED (card's) tensor.
    std::vector<std::pair<uint64_t, uint64_t>> c_spans, s_spans, o_spans;
};

/// How a segment's bytes must be transformed on the way from the pack to the arena.
enum class Conv : int {
    Copy = 0,      ///< the bytes are already the engine form
    Bf16High = 1,  ///< f32 container holding a bf16 value: take the high 16 bits
    ToF16 = 2,     ///< f32 container holding an f16 value: a real round-to-nearest-even conversion
    WidenF16 = 3,  ///< fp16 bits to be held as f32: EXACT, and the reason this path exists at all
};

/// One contiguous run of a tensor's source span with a single destination range and one transform.
///
/// THE SEGMENT LIST IS WHY THIS LOADER CAN CHANGE WIDTHS PLANE BY PLANE.  A quantized tensor as one
/// verbatim span is only correct when every plane is already in engine form - and for tensors whose scale
/// plane is fp16 in the file it must be f32 in the arena.
struct Seg {
    uint64_t src_off = 0;   ///< relative to the row's src_off
    uint64_t src_bytes = 0;
    uint64_t dst_off = 0;   ///< relative to the row's dst_off
    uint64_t dst_bytes = 0;
    Conv conv = Conv::Copy;
};

/// The index's lines, one at a time (the index is a small text file).
struct IndexText {
    std::vector<uint8_t> raw;
    size_t at = 0;
    bool open(const FileRange& r, std::string& err) {
        if (r.empty() || !read_range(r, raw, err)) { if (err.empty()) err = "no pack index"; return false; }
        return true;
    }
    /// The next line into `line` (NUL-terminated, at most `cap` - 1 bytes); false at the end.
    bool next(char* line, size_t cap) {
        if (at >= raw.size()) return false;
        size_t n = 0;
        while (at < raw.size() && raw[at] != '\n') { if (n + 1 < cap) line[n++] = (char) raw[at]; ++at; }
        if (at < raw.size()) ++at;
        line[n] = 0;
        return true;
    }
};

}  // namespace

PackFiles PackFiles::from(const ModelSources& s) {
    PackFiles p;
    p.index = s.dense_index;
    p.files[0] = s.dense;
    return p;
}

namespace {

bool plane_per_row(uint64_t plane_bytes, int64_t ne1, uint64_t& per_row) {
    if (plane_bytes == 0) { per_row = 0; return true; }
    if (ne1 <= 0 || plane_bytes % static_cast<uint64_t>(ne1) != 0) return false;
    per_row = plane_bytes / static_cast<uint64_t>(ne1);
    return true;
}

int64_t runs_total(const std::vector<std::pair<int64_t, int64_t>>& runs) {
    int64_t t = 0;
    for (const auto& r : runs) t += r.second;
    return t;
}

// The stored arena bytes for a row's planes under `runs` (dst units), or false with a reason.
bool sliced_dst_bytes(const char* name, int code_bits, int64_t ne1, uint64_t dst_bytes,
                      uint64_t codes_bytes, uint64_t scales_bytes, uint64_t offset_bytes,
                      const std::vector<std::pair<int64_t, int64_t>>& runs, uint64_t& out, std::string& err) {
    const int64_t total = runs_total(runs);
    if (total <= 0 || total > ne1) {
        err = std::string(name) + ": the row slice covers more rows than the tensor (or none)";
        return false;
    }
    if (code_bits != 0) {
        uint64_t pc = 0, ps = 0, po = 0;
        if (!plane_per_row(codes_bytes, ne1, pc) || !plane_per_row(scales_bytes, ne1, ps) ||
            !plane_per_row(offset_bytes, ne1, po)) {
            err = std::string(name) + ": the plane bytes are not row-uniform, no row slice exists";
            return false;
        }
        out = static_cast<uint64_t>(total) * (pc + ps + po);
        return true;
    }
    uint64_t pdst = 0;
    if (!plane_per_row(dst_bytes, ne1, pdst)) {
        err = std::string(name) + ": the row bytes are not uniform, no row slice exists";
        return false;
    }
    out = static_cast<uint64_t>(total) * pdst;
    return true;
}

/// One index line's stored bytes under `runs`, parsing the full 19-field row the same way `load` does.
bool line_sliced_dst(const char* line, const std::vector<std::pair<int64_t, int64_t>>& runs, uint64_t& out,
                     std::string& err) {
    char name[256] = {0};
    int kind = 0, i1 = 0;
    long long ne0 = 0, ne1 = 0;
    int code_bits = 0, code_bias = 0, group_elems = 0, codebook = 0, has_offset = 0, scales_fp16 = 0, act_kind = 0;
    unsigned long long src_off = 0, src_bytes = 0, dst_off = 0, dst_bytes = 0;
    unsigned long long codes_bytes = 0, scales_bytes = 0, offset_bytes = 0;
    const int n = std::sscanf(line,
                              "%255s %d %d %llu %llu %llu %llu %lld %lld %d %d %d %d %d %llu %llu %llu %d %d",
                              name, &i1, &kind, &src_off, &src_bytes, &dst_off, &dst_bytes, &ne0, &ne1,
                              &code_bits, &code_bias, &group_elems, &codebook, &has_offset, &codes_bytes,
                              &scales_bytes, &offset_bytes, &scales_fp16, &act_kind);
    if (n != 19) {
        err = std::string("row slice: could not parse an index row (") + std::to_string(n) + " of 19 fields)";
        return false;
    }
    return sliced_dst_bytes(name, code_bits, static_cast<int64_t>(ne1), dst_bytes, codes_bytes, scales_bytes,
                            offset_bytes, runs, out, err);
}

}  // namespace

bool WeightTable::pool_bytes(const PackFiles& pack, uint64_t& out, std::string& err,
                             const std::set<std::string>* skip, const RowSlices* slices) {
    const std::string path = pack.index.path;
    IndexText f;
    if (!f.open(pack.index, err)) { err = "cannot open " + path + (err.empty() ? "" : " (" + err + ")"); return false; }
    char line[1024];
    out = 0;
    uint64_t pool = 0, compact = 0;
    int align = 0;
    std::set<std::string> seen;   // ghost slice names: sizing must refuse what load refuses
    while (f.next(line, sizeof line)) {
        if (line[0] == '#') {
            unsigned long long p = 0;
            int a = 0, tensors = 0;
            if (std::sscanf(line, "# align %d pool %llu tensors %d", &a, &p, &tensors) == 3) { pool = p; align = a; }
            continue;
        }
        if (skip == nullptr && slices == nullptr) continue;
        char name[256] = {0};
        unsigned long long dst_bytes = 0, dummy = 0;
        int i1 = 0, i2 = 0;
        if (std::sscanf(line, "%255s %d %d %llu %llu %llu %llu", name, &i1, &i2, &dummy, &dummy, &dummy, &dst_bytes) != 7)
            continue;
        seen.insert(name);
        if (skip != nullptr && skip->count(name)) continue;
        const uint64_t a = align > 0 ? (uint64_t) align : 256;
        if (slices != nullptr && slices->count(name)) {
            uint64_t sbytes = 0;
            if (!line_sliced_dst(line, slices->at(name), sbytes, err)) return false;
            compact += (sbytes + a - 1) / a * a;
            continue;
        }
        compact += (dst_bytes + a - 1) / a * a;
    }
    if (pool == 0) { err = "no '# align ... pool ...' header in " + path; return false; }
    if (slices != nullptr)
        for (const auto& kv : *slices)
            if (!seen.count(kv.first)) { err = kv.first + ": the row slice names no index row"; return false; }
    out = (skip != nullptr || slices != nullptr) ? compact : pool;
    return true;
}

bool WeightTable::pool_bytes_by_layer(const PackFiles& pack, int64_t n_layers, std::vector<uint64_t>& per_layer,
                                      uint64_t& shared, std::string& err, const std::set<std::string>* skip,
                                      const RowSlices* slices) {
    IndexText f;
    if (!f.open(pack.index, err)) { err = "cannot open " + pack.index.path + (err.empty() ? "" : " (" + err + ")"); return false; }
    per_layer.assign((size_t) n_layers, 0);
    shared = 0;
    char line[1024];
    int align = 0;
    while (f.next(line, sizeof line)) {
        if (line[0] == '#') {
            unsigned long long p = 0;
            int a = 0, tensors = 0;
            if (std::sscanf(line, "# align %d pool %llu tensors %d", &a, &p, &tensors) == 3) align = a;
            continue;
        }
        char name[256] = {0};
        unsigned long long dst_bytes = 0, dummy = 0;
        int i1 = 0, i2 = 0;
        if (std::sscanf(line, "%255s %d %d %llu %llu %llu %llu", name, &i1, &i2, &dummy, &dummy, &dummy, &dst_bytes) != 7)
            continue;
        if (skip != nullptr && skip->count(name)) continue;
        const uint64_t a = align > 0 ? (uint64_t) align : 256;
        uint64_t eff_dst = dst_bytes;
        if (slices != nullptr && slices->count(name)) {
            if (!line_sliced_dst(line, slices->at(name), eff_dst, err)) return false;
        }
        const uint64_t bytes = (eff_dst + a - 1) / a * a;
        const long l = std::strncmp(name, "blk.", 4) == 0 ? std::strtol(name + 4, nullptr, 10) : -1;
        if (l >= 0 && l < n_layers) per_layer[(size_t) l] += bytes;
        else shared += bytes;
    }
    return true;
}

bool WeightTable::index_code_bits(const PackFiles& pack, const std::string& name, int& code_bits,
                                  std::string& err) {
    IndexText f;
    if (!f.open(pack.index, err)) { err = "cannot open " + pack.index.path; return false; }
    char line[1024];
    code_bits = -1;
    while (f.next(line, sizeof line)) {
        if (line[0] == '#') continue;
        char row[256] = {0};
        unsigned long long dummy = 0;
        long long ne = 0;
        int i1 = 0, i2 = 0, bits = 0;
        // the first ten fields of the row `load` parses: name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits
        if (std::sscanf(line, "%255s %d %d %llu %llu %llu %llu %lld %lld %d", row, &i1, &i2, &dummy, &dummy, &dummy,
                        &dummy, &ne, &ne, &bits) != 10)
            continue;
        if (name == row) { code_bits = bits; break; }
    }
    return true;
}

bool WeightTable::load(const PackFiles& pack, void* arena_base, uint64_t arena_bytes, std::string& err,
                       const std::set<std::string>* skip, bool report_progress, const RowSlices* slices) {
    IndexText idx;
    if (!idx.open(pack.index, err)) { err = "cannot open " + pack.index.path; return false; }

    uint64_t pool = 0;
    int align = 0;
    std::vector<IndexRow> rows;
    char line[1024];
    while (idx.next(line, sizeof line)) {
        if (line[0] == '#') {
            unsigned long long p = 0;
            int a = 0, t = 0;
            if (std::sscanf(line, "# align %d pool %llu tensors %d", &a, &p, &t) == 3) { pool = p; align = a; }
            continue;
        }
        IndexRow r;
        char name[256] = {0};
        int kind = 0;
        // The format is written when the model file is built and is space-separated with no quoting; a name with a
        // space in it would break the parse, which is why the parser REFUSES rather than taking what it got.
        const int n = std::sscanf(line,
                                  "%255s %d %d %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNd64 " %" SCNd64
                                  " %d %d %d %d %d %" SCNu64 " %" SCNu64 " %" SCNu64 " %d %d",
                                  name, &r.file, &kind, &r.src_off, &r.src_bytes, &r.dst_off, &r.dst_bytes,
                                  &r.ne0, &r.ne1, &r.code_bits, &r.code_bias, &r.group_elems, &r.codebook,
                                  &r.has_offset, &r.codes_bytes, &r.scales_bytes, &r.offset_bytes,
                                  &r.scales_fp16, &r.act_kind);
        if (n != 19) {
            // A malformed index line is a WRONG BYTE OFFSET waiting to happen, and a wrong offset decodes to
            // a plausible weight.  Refuse loudly instead of loading the rows that happened to parse.
            err = "dense index: could not parse a row (" + std::to_string(n) + " of 19 fields)";
            return false;
        }
        r.name = name;
        // 16-bit floats are stored as they are in the GGUF:
        // index kinds 4 (BF16) and 5 (F16) are copied and take the engine forms of kinds 1 and 3
        if (kind == 4 || kind == 5) {
            r.raw16 = true;
            kind = kind == 4 ? (int) WeightKind::Bf16InF32 : (int) WeightKind::F16InF32;
        }
        r.kind = (WeightKind) kind;
        rows.push_back(r);
    }

    if (pool == 0 || rows.empty()) { err = "the dense index has no header or no rows"; return false; }

    // Row slices: rewrite the named rows to their stored spans BEFORE any sizing or placement decision, so
    // compaction, verification and the published WeightRefs all describe the card's tensor.
    if (slices != nullptr) {
        std::set<std::string> seen;
        for (auto& r : rows) {
            const auto it = slices->find(r.name);
            if (it == slices->end()) continue;
            seen.insert(r.name);
            if (it->second.empty() || runs_total(it->second) > r.ne1) {
                err = r.name + ": the row slice covers more rows than the tensor (or none)";
                return false;
            }
            {   // per-run bounds + disjointness: the header promises a hard error on a misaligned or over-large
                // run list - a single run past ne1 with a small total, or overlapping runs, used to slip through
                // and read past the tensor into the next tensor's source bytes
                auto rs = it->second;
                std::sort(rs.begin(), rs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                uint64_t end = 0;
                for (const auto& run : rs) {
                    if (run.first < 0 || run.second <= 0 ||
                        (uint64_t) run.first + (uint64_t) run.second > (uint64_t) r.ne1) {
                        err = r.name + ": a row-slice run is outside the tensor";
                        return false;
                    }
                    if ((uint64_t) run.first < end) { err = r.name + ": overlapping row-slice runs"; return false; }
                    end = (uint64_t) run.first + (uint64_t) run.second;
                }
            }
            uint64_t stored_dst = 0;
            if (r.code_bits != 0) {
                uint64_t pc = 0, ps = 0, po = 0;
                if (!plane_per_row(r.codes_bytes, r.ne1, pc) || !plane_per_row(r.scales_bytes, r.ne1, ps) ||
                    !plane_per_row(r.offset_bytes, r.ne1, po)) {
                    err = r.name + ": the plane bytes are not row-uniform, no row slice exists";
                    return false;
                }
                const uint64_t scale_src_pr = r.scales_fp16 ? ps / 2 : ps;
                const uint64_t scale_src_base = r.codes_bytes;
                const uint64_t offset_src_base = r.codes_bytes + (r.scales_fp16 ? r.scales_bytes / 2 : r.scales_bytes);
                for (const auto& run : it->second) {
                    if (run.first < 0 || run.second <= 0) { err = r.name + ": a bad row-slice run"; return false; }
                    if (pc) r.c_spans.push_back({static_cast<uint64_t>(run.first) * pc, static_cast<uint64_t>(run.second) * pc});
                    if (ps)
                        r.s_spans.push_back({scale_src_base + static_cast<uint64_t>(run.first) * scale_src_pr,
                                             static_cast<uint64_t>(run.second) * scale_src_pr});
                    if (po)
                        r.o_spans.push_back({offset_src_base + static_cast<uint64_t>(run.first) * po,
                                             static_cast<uint64_t>(run.second) * po});
                }
                r.codes_bytes = static_cast<uint64_t>(runs_total(it->second)) * pc;
                r.scales_bytes = static_cast<uint64_t>(runs_total(it->second)) * ps;
                r.offset_bytes = static_cast<uint64_t>(runs_total(it->second)) * po;
                uint64_t src = 0;
                for (const auto& sp : r.c_spans) src += sp.second;
                for (const auto& sp : r.s_spans) src += sp.second;
                for (const auto& sp : r.o_spans) src += sp.second;
                r.src_bytes = src;
                stored_dst = r.codes_bytes + r.scales_bytes + r.offset_bytes;
            } else {
                uint64_t psrc = 0, pdst = 0;
                if (!plane_per_row(r.src_bytes, r.ne1, psrc) || !plane_per_row(r.dst_bytes, r.ne1, pdst)) {
                    err = r.name + ": the row bytes are not uniform, no row slice exists";
                    return false;
                }
                for (const auto& run : it->second) {
                    if (run.first < 0 || run.second <= 0) { err = r.name + ": a bad row-slice run"; return false; }
                    r.c_spans.push_back({static_cast<uint64_t>(run.first) * psrc, static_cast<uint64_t>(run.second) * psrc});
                }
                r.src_bytes = static_cast<uint64_t>(runs_total(it->second)) * psrc;
                stored_dst = static_cast<uint64_t>(runs_total(it->second)) * pdst;
            }
            r.dst_bytes = stored_dst;
            r.ne1 = runs_total(it->second);
        }
        for (const auto& s : *slices)
            if (!seen.count(s.first)) {
                err = "row slice names " + s.first + ", which the index does not have";
                return false;
            }
    }

    // Plan v0.3 P1: a skip set compacts the arena.  Kept rows are re-placed in index order at the index's
    // alignment; skipped rows keep their metadata and get no bytes.
    std::vector<bool> skipped(rows.size(), false);
    if (skip != nullptr || slices != nullptr) {
        const uint64_t a = align > 0 ? (uint64_t) align : 256;
        uint64_t at = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (skip != nullptr && skip->count(rows[i].name)) { skipped[i] = true; continue; }
            rows[i].dst_off = at;
            at += (rows[i].dst_bytes + a - 1) / a * a;
        }
        pool = at;
    }
    if (arena_bytes < pool) {
        err = "arena is " + std::to_string(arena_bytes) + " B but the index needs " + std::to_string(pool);
        return false;
    }

    // Two pinned staging buffers, both reused.  Pinned because a 5 GB pageable upload spends its time in the
    // driver copying through a bounce buffer, and bounded by CHUNK rather than by the largest tensor.
    //
    // STAGE_OUT IS NOT OPTIONAL.  The first version of this loop converted the values straight into the
    // destination pointer - which is DEVICE memory - from host code.  A host write to a device address is
    // undefined behaviour, and on this driver it does not fault: it corrupts whatever the address maps to in
    // the host's own space, which is nothing at all, so the tensor would have loaded as the UNCONVERTED pack
    // bytes and every BF16 weight would have been read as 32-bit garbage.  Converted data goes to a host
    // buffer and crosses the bus once.
    //
    // STAGE_OUT IS TWICE THE SIZE OF A READ because a conversion can now GROW: widening an fp16 scale plane to f32
    // doubles it.  Sized at CHUNK it would have written 8 MiB past the end of a pinned allocation for every
    // one of the 90 widened tensors - a heap corruption that would have been blamed on whatever ran next.
    void* stage_out = nullptr;
    if (cudaHostAlloc(&stage_out, CHUNK * 2, cudaHostAllocDefault) != cudaSuccess) {
        err = "cudaHostAlloc for the staging buffers failed";
        return false;
    }

    uint8_t* dst_base = (uint8_t*) arena_base;
    basalt::core::DirectUploader reader;   // unbuffered reads into pinned staging
    int cur_file = -1;
    std::string cur_path;
    uint64_t cur_base = 0, cur_size = 0;   // the current file's range: reads are relative to it
    table_.clear();
    report_ = LoadReport{};

    int load_dev = 0;
    if (cudaGetDevice(&load_dev) != cudaSuccess) load_dev = 0;
    // a temporary load (staging that is dropped again) is not part of the model's bytes
    std::optional<LoadProgress> progress;
    if (report_progress) progress.emplace(LoadProgress::cuda(load_dev), pool);
    const double t_read0 = now_ms();
    double upload_ms = 0;
    for (size_t row_i = 0; row_i < rows.size(); ++row_i) {
        const IndexRow& r = rows[row_i];
        if (skipped[row_i]) {
            WeightRef wr;
            wr.data = nullptr;
            wr.resident = false;
            wr.bytes = r.dst_bytes;
            wr.ne0 = r.ne0;
            wr.ne1 = r.ne1;
            wr.elements = r.ne0 * (r.ne1 > 0 ? r.ne1 : 1);
            wr.kind = r.kind;
            wr.code_bits = r.code_bits;
            wr.code_bias = r.code_bias;
            wr.group_elems = r.group_elems;
            wr.codebook_iq4nl = r.codebook != 0;
            wr.has_offset = r.has_offset != 0;
            wr.codes_bytes = r.codes_bytes;
            wr.scales_bytes = r.scales_bytes;
            wr.offset_bytes = r.offset_bytes;
            wr.src_off = r.src_off;
            wr.src_bytes = r.src_bytes;
            wr.file_id = r.file;
            wr.scales_fp16 = r.scales_fp16 != 0;
            wr.act_kind = r.act_kind;
            table_[r.name] = wr;
            ++report_.tensors;
            continue;
        }
        if (r.code_bits != 0 && r.dst_bytes == 0) {
            // a row that carries a shape only - the GGUF form must serve it
            err = r.name + ": the model file holds the tensor only in its GGUF form";
            cudaFreeHost(stage_out);
            return false;
        }
        if (r.file != cur_file) {
            cur_file = r.file;
            if (r.file < 0 || r.file > 3 || pack.files[r.file].empty()) {
                err = r.name + ": the index names file " + std::to_string(r.file) + ", which this pack does not have";
                cudaFreeHost(stage_out);
                return false;
            }
            cur_base = pack.files[r.file].offset;
            cur_size = pack.files[r.file].bytes;
            cur_path = pack.files[r.file].path;
        }

        // ---- the segment list: what this tensor's bytes are, plane by plane, in both forms
        std::vector<Seg> segs;
        bool bad = false;
        uint64_t sum_src = 0, sum_dst = 0;
        // A sliced row: the spans, plane by plane, become the segments directly (the plane fields already
        // hold the stored sizes; the spans hold the file offsets and lengths, relative to src_off).
        const bool sliced_row = !r.c_spans.empty() || !r.s_spans.empty() || !r.o_spans.empty();
        if (sliced_row && r.code_bits != 0) {
            // each plane's spans tile that plane in run order: a per-plane cursor, NOT a fixed plane base -
            // with >1 run a fixed base overwrites the earlier spans inside the plane (the byte totals still
            // match, so the sum guard below cannot see it). One run: identical segments to the fixed base.
            uint64_t at_c = 0, at_s = 0, at_o = 0;
            for (const auto& sp : r.c_spans) {
                segs.push_back(Seg{sp.first, sp.second, at_c, sp.second, Conv::Copy});
                at_c += sp.second;
            }
            for (const auto& sp : r.s_spans) {
                const uint64_t dlen = sp.second * (r.scales_fp16 ? 2 : 1);
                segs.push_back(Seg{sp.first, sp.second, r.codes_bytes + at_s, dlen,
                                   r.scales_fp16 ? Conv::WidenF16 : Conv::Copy});
                at_s += dlen;
            }
            for (const auto& sp : r.o_spans) {
                segs.push_back(Seg{sp.first, sp.second, r.codes_bytes + r.scales_bytes + at_o, sp.second, Conv::Copy});
                at_o += sp.second;
            }
        } else if (sliced_row) {
            const uint64_t dst_per_src = r.src_bytes > 0 ? r.dst_bytes * 1000 / r.src_bytes : 1000;
            uint64_t at = 0;
            for (const auto& sp : r.c_spans) {
                const uint64_t dlen = sp.second * dst_per_src / 1000;
                Conv c = Conv::Copy;
                if (!r.raw16) {
                    if (r.kind == WeightKind::Bf16InF32) c = Conv::Bf16High;
                    else if (r.kind == WeightKind::F16InF32) c = Conv::ToF16;
                }
                segs.push_back(Seg{sp.first, sp.second, at, dlen, c});
                at += dlen;
            }
        } else if (r.codes_bytes != 0) {
            // A quantized tensor: codes, then scales, then the optional offset plane, each contiguous in the
            // source span in that order (the converter verifies the contiguity and refuses otherwise).
            segs.push_back(Seg{0, r.codes_bytes, 0, r.codes_bytes, Conv::Copy});
            if (r.scales_bytes) {
                const uint64_t src_scale_bytes = r.scales_fp16 ? r.scales_bytes / 2 : r.scales_bytes;
                if (r.scales_fp16 && (r.scales_bytes % 2 || src_scale_bytes * 2 != r.scales_bytes)) {
                    err = r.name + ": an fp16 scale plane of an odd byte count";
                    bad = true;
                }
                segs.push_back(Seg{r.codes_bytes, src_scale_bytes, r.codes_bytes, r.scales_bytes,
                                   r.scales_fp16 ? Conv::WidenF16 : Conv::Copy});
            }
            if (r.offset_bytes) {
                const uint64_t after = r.codes_bytes + (r.scales_fp16 ? r.scales_bytes / 2 : r.scales_bytes);
                segs.push_back(Seg{after, r.offset_bytes, r.codes_bytes + r.scales_bytes, r.offset_bytes,
                                   Conv::Copy});
            }
        } else {
            // Everything else is one run whose transform is the row's kind.  The two promoted types shrink
            // 4 B/elem to 2; the rest are already the engine form.
            Conv c = Conv::Copy;
            if (r.raw16) c = Conv::Copy;
            else if (r.kind == WeightKind::Bf16InF32) c = Conv::Bf16High;
            else if (r.kind == WeightKind::F16InF32) c = Conv::ToF16;
            const uint64_t elems = (uint64_t) r.ne0 * (uint64_t) (r.ne1 > 0 ? r.ne1 : 1);
            if (c != Conv::Copy && (r.src_bytes != elems * 4 || r.dst_bytes != elems * 2)) {
                // A promoted type is 4 B/elem in and 2 B/elem out, and nothing else.  If those two numbers do
                // not say so, the shape and the byte counts disagree and the element count this loop would
                // use to walk the source is not the element count the pack wrote.
                char buf[320];
                std::snprintf(buf, sizeof buf,
                              "%s: a promoted tensor of %llu elements is %llu B in and %llu B out, not %llu/%llu",
                              r.name.c_str(), (unsigned long long) elems, (unsigned long long) r.src_bytes,
                              (unsigned long long) r.dst_bytes, (unsigned long long) (elems * 4),
                              (unsigned long long) (elems * 2));
                err = buf;
                bad = true;
            }
            segs.push_back(Seg{0, r.src_bytes, 0, r.dst_bytes, c});
        }
        for (size_t i = 0; i < segs.size(); ++i) { sum_src += segs[i].src_bytes; sum_dst += segs[i].dst_bytes; }
        // The segment list is this loop's own idea of the layout, and it is the ONLY thing standing between a
        // mis-sized plane and a wrong byte offset.  It is checked against both recorded totals, not trusted.
        if (!bad && (sum_src != r.src_bytes || sum_dst != r.dst_bytes)) {
            char buf[360];
            std::snprintf(buf, sizeof buf,
                          "%s: the segments cover %llu/%llu B but the index says %llu/%llu - the plane layout "
                          "this loader built is not the one the index describes",
                          r.name.c_str(), (unsigned long long) sum_src, (unsigned long long) sum_dst,
                          (unsigned long long) r.src_bytes, (unsigned long long) r.dst_bytes);
            err = buf;
            bad = true;
        }
        if (bad) { cudaFreeHost(stage_out); return false; }

        for (size_t si = 0; si < segs.size(); ++si) {
            const Seg& s = segs[si];
            const bool widening = (s.conv == Conv::WidenF16);
            if (widening && (s.src_bytes & 1ull)) {
                err = r.name + ": an fp16 plane with an odd source byte count";
                cudaFreeHost(stage_out);
                return false;
            }
            uint64_t done = 0;
            while (done < s.src_bytes) {
                uint64_t n = std::min(CHUNK, s.src_bytes - done);
                // WIDENING NEVER SPLITS AN ELEMENT.  CHUNK is even, so a full chunk cannot, but the tail of a
                // plane whose length is odd would - and half an fp16 is a plausible-looking scale.
                if (widening && (n & 1ull)) {
                    if (n == 1) { err = r.name + ": an fp16 plane ending on a half element"; cudaFreeHost(stage_out); return false; }
                    n -= 1;
                }
                if (cur_size != 0 && r.src_off + s.src_off + done + n > cur_size) {
                    err = r.name + ": runs past the end of its section";
                    cudaFreeHost(stage_out);
                    return false;
                }
                const uint8_t* src = reader.staged(cur_path, cur_base + r.src_off + s.src_off + done, n, err);
                if (src == nullptr) {
                    err = r.name + ": " + err;
                    cudaFreeHost(stage_out);
                    return false;
                }
                const void* host_src = src;
                uint64_t out_bytes = n;
                uint64_t out_at = s.dst_off + done;

                switch (s.conv) {
                    case Conv::Copy:
                        break;
                    case Conv::Bf16High: {
                        // THE HIGH 16 BITS ARE THE BF16 ENCODING, exactly - not a rounding.  A bf16 value is
                        // the top half of its f32 image, and the pack promoted it without changing it, so
                        // this is a copy with a stride.
                        const uint64_t elems = n / 4;
                        out_bytes = elems * 2;
                        uint8_t* o = (uint8_t*) stage_out;
                        for (uint64_t i = 0; i < elems; ++i) {
                            uint32_t v;
                            std::memcpy(&v, src + i * 4, 4);
                            const uint16_t h = (uint16_t) (v >> 16);
                            std::memcpy(o + i * 2, &h, 2);
                        }
                        host_src = stage_out;
                        out_at = s.dst_off + done / 2;
                        break;
                    }
                    case Conv::ToF16: {
                        const uint64_t elems = n / 4;
                        out_bytes = elems * 2;
                        uint8_t* o = (uint8_t*) stage_out;
                        for (uint64_t i = 0; i < elems; ++i) {
                            float v;
                            std::memcpy(&v, src + i * 4, 4);
                            const uint16_t h = basalt::kernels::f16_from_f32(v);
                            std::memcpy(o + i * 2, &h, 2);
                        }
                        host_src = stage_out;
                        out_at = s.dst_off + done / 2;
                        break;
                    }
                    case Conv::WidenF16: {
                        // fp16 bits -> f32.  EXACT: every fp16 value is an f32 value, including the
                        // subnormals, so this cannot change a single weight - which is why the engine may do
                        // it at all, and why doing it costs a little memory rather than a second kernel family.
                        const uint64_t elems = n / 2;
                        out_bytes = elems * 4;
                        float* o = (float*) stage_out;
                        for (uint64_t i = 0; i < elems; ++i) {
                            uint16_t h;
                            std::memcpy(&h, src + i * 2, 2);
                            o[i] = basalt::kernels::f32_from_f16(h);
                        }
                        host_src = stage_out;
                        out_at = s.dst_off + done * 2;
                        break;
                    }
                }

                const double u0 = now_ms();
                if (cudaMemcpy(dst_base + r.dst_off + out_at, host_src, out_bytes, cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
                    err = "cudaMemcpy failed for " + r.name;
                    cudaFreeHost(stage_out);
                    return false;
                }
                upload_ms += now_ms() - u0;
                if (progress) progress->add(out_bytes);
                done += n;
            }
        }

        WeightRef wr;
        wr.data = dst_base + r.dst_off;
        wr.bytes = r.dst_bytes;
        wr.ne0 = r.ne0;
        wr.ne1 = r.ne1;
        wr.elements = r.ne0 * (r.ne1 > 0 ? r.ne1 : 1);
        wr.kind = r.kind;
        wr.code_bits = r.code_bits;
        wr.code_bias = r.code_bias;
        wr.group_elems = r.group_elems;
        wr.codebook_iq4nl = r.codebook != 0;
        wr.has_offset = r.has_offset != 0;
        wr.codes_bytes = r.codes_bytes;
        wr.scales_bytes = r.scales_bytes;
        wr.offset_bytes = r.offset_bytes;
        wr.src_off = r.src_off;
        wr.src_bytes = r.src_bytes;
        wr.file_id = r.file;
        wr.scales_fp16 = r.scales_fp16 != 0;
        wr.act_kind = r.act_kind;
        table_[r.name] = wr;

        ++report_.tensors;
        if ((r.kind == WeightKind::Bf16InF32 || r.kind == WeightKind::F16InF32) && !r.raw16) {
            ++report_.re_rounded;
            report_.bytes_saved += r.src_bytes - r.dst_bytes;
        }
    }
    cudaFreeHost(stage_out);
    if (progress) progress->finish();

    report_.arena_bytes = pool;
    report_.read_ms = now_ms() - t_read0 - upload_ms;
    report_.upload_ms = upload_ms;
    if (cudaDeviceSynchronize() != cudaSuccess) { err = "cudaDeviceSynchronize after the load failed"; return false; }
    return true;
}

const WeightRef* WeightTable::find(const std::string& name) const {
    const auto it = table_.find(name);
    return it == table_.end() ? nullptr : &it->second;
}

bool WeightTable::rebind(const std::string& name, const void* data) {
    const auto it = table_.find(name);
    if (it == table_.end() || data == nullptr) return false;
    it->second.data = data;
    it->second.resident = true;
    return true;
}

}  // namespace basalt::core
