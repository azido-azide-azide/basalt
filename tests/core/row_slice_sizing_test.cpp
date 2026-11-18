// tests/core/row_slice_sizing_test.cpp - WeightTable::RowSlices, host only: the synthetic index's sliced
// arena sizes are exactly the per-plane row arithmetic (quantized planes and raw/promoted rows), sizing by
// layer agrees with the total, and the refusals refuse: over-long run lists, names the index lacks, and a
// tensor whose planes are not row-uniform.
#include "basalt/core/weights.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using basalt::core::PackFiles;
using basalt::core::WeightTable;

namespace fs = std::filesystem;

namespace {
int failures = 0;
void ok(bool cond, const std::string& what) {
    std::printf("%s %s\n", cond ? "ok  " : "FAIL", what.c_str());
    if (!cond) ++failures;
}
uint64_t al(uint64_t b) { return (b + 255) / 256 * 256; }   // the index's align 256

uint64_t pool_of(const PackFiles& p, const WeightTable::RowSlices* slices, std::string& err, bool expect_ok) {
    uint64_t pool = 0;
    const bool r = WeightTable::pool_bytes(p, pool, err, nullptr, slices);
    if (r != expect_ok) {
        std::printf("     (%s: %s)\n", expect_ok ? "wanted ok" : "wanted refuse", err.c_str());
        return expect_ok ? pool : 0;
    }
    return pool;
}
}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "row_slice_sizing_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    {
        // a small dense file exists so a caller COULD load; pool_bytes only reads the index
        std::ofstream dense(dir / "dense.bin", std::ios::binary);
        dense.write(std::string(1 << 16, '\x5a').data(), 1 << 16);
        std::ofstream idx(dir / "index.txt");
        idx << "# align 256 pool " << 1000000 << " tensors 4\n";
        // A: quantized, 16 rows x 32 cols, 4bpv codes (256 B), 32-groups scales 4 B (64 B), no offsets.
        idx << "blk.0.attn_qkv.weight 0 0 0 320 0 320 32 16 4 0 32 0 0 256 64 0 0 0\n";
        // B: raw BF16 (index kind 4): 8 rows x 8 cols x 2 B = 128.
        idx << "blk.0.attn_q.weight 0 4 320 128 0 128 8 8 0 0 0 0 0 0 0 0 0 0\n";
        // C: BF16 promoted to F32 in the file (index kind 1): 4 rows x 4 cols x 4 B src -> 2 B dst = 32.
        idx << "blk.0.hc_x.weight 0 1 448 64 0 32 4 4 0 0 0 0 0 0 0 0 0 0\n";
        // D: a non-row-uniform quantized row (260 codes for 16 rows) - slices must refuse it.
        idx << "blk.1.bad.weight 0 0 512 324 0 324 32 16 4 0 32 0 0 260 64 0 0 0\n";
    }
    const PackFiles p = [&] {
        PackFiles f;
        f.index = basalt::core::FileRange((dir / "index.txt").string());
        f.files[0] = basalt::core::FileRange((dir / "dense.bin").string());
        return f;
    }();

    std::string err;
    const uint64_t full = pool_of(p, nullptr, err, true);
    ok(full == 1000000, "without slices the pool is the header's");

    WeightTable::RowSlices slices;
    slices["blk.0.attn_qkv.weight"] = {{0, 4}, {8, 4}};    // 8 rows: 128 codes + 32 scales
    slices["blk.0.attn_q.weight"] = {{4, 4}};              // 4 rows: 64 B
    const uint64_t sl = pool_of(p, &slices, err, true);
    ok(sl == al(160) + al(64) + al(32) + al(324), "a slices-only load compacts the arena to the row arithmetic");
    std::vector<uint64_t> per_layer;
    uint64_t shared = 0;
    ok(WeightTable::pool_bytes_by_layer(p, 2, per_layer, shared, err, nullptr, &slices),
       "by-layer sizing accepts the slices");
    const uint64_t want0 = al(160) + al(64) + al(32);   // sliced A, sliced B, whole C on layer 0
    const uint64_t want1 = al(324);                     // untouched D
    ok(per_layer.size() == 2 && per_layer[0] == want0 && per_layer[1] == want1,
       "the per-layer bytes are the row arithmetic exactly");

    {
        WeightTable::RowSlices over;
        over["blk.0.attn_qkv.weight"] = {{0, 8}, {8, 9}};   // 17 rows on a 16-row tensor
        ok(!WeightTable::pool_bytes(p, shared, err, nullptr, &over), "a run list over the row count is refused");
    }

    WeightTable::RowSlices ghost;
    ghost["blk.7.no_there.weight"] = {{0, 2}};
    ok(!WeightTable::pool_bytes(p, shared, err, nullptr, &ghost) || !err.empty(),
       "a slice naming no index row is refused");

    WeightTable::RowSlices bad_plane;
    bad_plane["blk.1.bad.weight"] = {{0, 2}};
    ok(!WeightTable::pool_bytes(p, shared, err, nullptr, &bad_plane) || !err.empty(),
       "a non-row-uniform tensor refuses the slice");

    fs::remove_all(dir);
    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
