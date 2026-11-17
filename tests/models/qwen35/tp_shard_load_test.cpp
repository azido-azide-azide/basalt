// tests/models/qwen35/tp_shard_load_test.cpp - the tensor-parallel load of the real 27B pack: the adapter slices
// every layer tensor per card (rank 0 and rank 1 together hold each payload exactly once - the fragments of both
// cards, scattered back to their source offsets, rebuild the pack byte for byte), rank 0 alone holds the head,
// embedding and drafter, and the bytes the load reports match what the tensor table says they must.  A load with
// --tp and --diag-split-at together refuses.  Needs a 27B pack (argument or BASALT_QWEN35_PACK) and two GPUs
// (the load puts a card's slices on each); exits 77 without them.
#include "basalt/core/model_file.hpp"
#include "basalt/dense/tp.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/platform/file.hpp"
#include "models/qwen35/adapter.hpp"
#include "models/qwen35/meta.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace basalt;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}
[[noreturn]] void die(const char* what, const std::string& msg) {
    std::fprintf(stderr, "%s: %s\n", what, msg.c_str());
    std::exit(1);
}

dense::Geometry geometry_of(const GgufFile& gg, std::string& err) {
    return models::qwen35::geometry(gg, err);
}

// one card's slices of one tensor, as the load plans them: the fragments of its payload
struct CardPlan {
    std::vector<dense::tp::Frag> frags;   // into the tensor's full payload
    bool sliced = false;
};

CardPlan plan_tensor(uint32_t type, int64_t rows, int64_t cols, const std::string& name, const dense::Geometry& g,
                     const dense::tp::RankPlan& p) {
    CardPlan c;
    const int64_t H = g.n_embd, AQ = (int64_t) g.n_head * g.head_dim, KV = (int64_t) g.n_kv * g.head_dim;
    const int64_t C = g.conv_channels(), VD = g.value_width(), HV = g.gdn_v_heads, FF = g.n_ff;
    const auto suf = [&](const char* e) { return name.size() >= strlen(e) && name.compare(name.size() - strlen(e), strlen(e), e) == 0; };
    std::vector<core::GdnRun> runs;
    bool cols_axis = false;
    bool hit = true;
    if (name.starts_with("blk.")) {
        if (suf("attn_qkv.weight")) { runs = p.qkv_rows(); }
        else if (suf("ssm_conv1d.weight")) { runs = p.qkv_rows(); }
        else if (suf("attn_gate.weight")) { runs = p.v_rows(g.gdn_state); }
        else if (suf("ssm_alpha.weight") || suf("ssm_beta.weight")) { runs = p.v_rows(1); }
        else if (suf("ssm_dt.bias") || suf("ssm_a")) { runs = p.v_rows(1); }
        else if (suf("ssm_out.weight")) { runs = p.out_cols(); cols_axis = true; }
        else if (suf("attn_q.weight")) { runs = p.q_rows(); }
        else if (suf("attn_k.weight") || suf("attn_v.weight")) { runs = p.kv_rows(); }
        else if (suf("attn_output.weight")) { runs = p.o_cols(); cols_axis = true; }
        else if (suf("ffn_gate.weight") || suf("ffn_up.weight")) { runs = p.ffn_rows(); }
        else if (suf("ffn_down.weight")) { runs = p.ffn_cols(); cols_axis = true; }
        else hit = false;
    } else
        hit = false;
    if (!hit) return c;   // shared or rank-0-only: whole
    c.frags = dense::tp::payload_fragments(type, rows, cols, runs, cols_axis);
    c.sliced = !c.frags.empty();
    return c;
}

int64_t axis_of(const std::vector<core::GdnRun>& runs) {
    int64_t n = 0;
    for (const core::GdnRun& r : runs) n += r.len;
    return n;
}

// the tensor's dimensions in kernels' order (rows of `cols` values)
bool dims_of(const std::string& name, const dense::Geometry& g, int64_t& rows, int64_t& cols) {
    const int64_t H = g.n_embd, AQ = (int64_t) g.n_head * g.head_dim, KV = (int64_t) g.n_kv * g.head_dim;
    const auto suf = [&](const char* e) { return name.size() >= strlen(e) && name.compare(name.size() - strlen(e), strlen(e), e) == 0; };
    if (!name.starts_with("blk.")) return false;
    if (suf("attn_qkv.weight")) { rows = g.conv_channels(); cols = H; }
    else if (suf("ssm_conv1d.weight")) { rows = g.conv_channels(); cols = g.gdn_conv; }
    else if (suf("attn_gate.weight")) { rows = g.value_width(); cols = H; }
    else if (suf("ssm_alpha.weight") || suf("ssm_beta.weight")) { rows = g.gdn_v_heads; cols = H; }
    else if (suf("ssm_dt.bias") || suf("ssm_a")) { rows = g.gdn_v_heads; cols = 1; }
    else if (suf("ssm_out.weight")) { rows = H; cols = g.value_width(); }
    else if (suf("attn_q.weight")) { rows = 2 * AQ; cols = H; }
    else if (suf("attn_k.weight") || suf("attn_v.weight")) { rows = KV; cols = H; }
    else if (suf("attn_output.weight")) { rows = H; cols = AQ; }
    else if (suf("ffn_gate.weight") || suf("ffn_up.weight")) { rows = g.n_ff; cols = H; }
    else if (suf("ffn_down.weight")) { rows = H; cols = g.n_ff; }
    else return false;
    return true;
}

int64_t axis_len(const std::vector<core::GdnRun>& runs) {
    int64_t n = 0;
    for (const core::GdnRun& r : runs) n += r.len;
    return n;
}

bool on_card_like(const std::string& name, dense::Drafter drafter, int rank) {
    if (name.starts_with("blk.")) return true;
    return rank == 0 && (name.starts_with("output") || (drafter == dense::Drafter::Block && name.starts_with("dflash.")));
}

// run one load, check the reported bytes and both cards' fragment coverage of every layer tensor
void try_load(const model::ModelAdapter& a, const std::string& pack, bool draft) {
    model::LoadRequest req;
    req.paths = {pack};
    req.max_context = 8192;
    req.sessions = 1;
    req.tp = 2;
    req.draft = draft;
    req.spec = draft ? 16 : 0;
    std::unique_ptr<model::LoadedModel> loaded;
    const model::Result r = a.load(req, loaded);
    const std::string tag = draft ? " with the block drafter" : "";
    if (r.ok()) {
        check(false, "a --tp 2 load stops where the step ends (not wired yet)" + tag);
        return;
    }
    const bool got = r.message.find("tensor-parallel weights loaded") != std::string::npos;
    check(got, "the --tp 2 load placed both cards and reported it" + tag + " (" + r.message + ")");
    if (!got) return;
    // rank0 MiB and rank1 MiB out of the message
    const auto dig = [&](const std::string& key) -> long long {
        size_t at = r.message.find(key);
        if (at == std::string::npos) return -1;
        at += key.size();
        size_t b = r.message.find_first_of("0123456789", at);
        size_t e = r.message.find(" MiB", b);
        return b == std::string::npos || e == std::string::npos ? -1 : std::atoll(r.message.substr(b, e - b).c_str());
    };
    const long long rep0 = dig("rank0 "), rep1 = dig("rank1 ");
    check(rep0 > 0 && rep1 > 0, "the message carries both ranks' MiB (" + std::to_string(rep0) + ", " + std::to_string(rep1) + ")");

    // expected bytes from the tensor table, and the two cards' fragments per tensor
    std::string err;
    const GgufFile gg(pack);
    const dense::Geometry g = geometry_of(gg, err);
    if (!err.empty()) die("geometry", err);
    const dense::tp::RankPlan p0 = dense::tp::rank_plan(g, 2, 0, err), p1 = dense::tp::rank_plan(g, 2, 1, err);
    const auto drafter = draft ? dense::Drafter::Block : dense::Drafter::None;
    uint64_t want0 = 0, want1 = 0;
    int fd = platform::open_fd_utf8(pack);
    if (fd < 0) die("open", pack);
    int sliced_tensors = 0;
    for (const TensorInfo& t : gg.tensors()) {
        if (t.name.starts_with("mtp.")) continue;   // the pack's MTP never runs while the block drafter is asked for
        int64_t rows = 0, cols = 0;
        const bool matrix = dims_of(t.name, g, rows, cols);
        CardPlan c, d;
        if (matrix) {
            c = plan_tensor(t.type, rows, cols, t.name, g, p0);
            d = plan_tensor(t.type, rows, cols, t.name, g, p1);
        }
        if (on_card_like(t.name, drafter, 0)) want0 += c.sliced ? dense::tp::shard_bytes(c.frags) : tensor_payload_bytes(t);
        if (on_card_like(t.name, drafter, 1)) want1 += d.sliced ? dense::tp::shard_bytes(d.frags) : tensor_payload_bytes(t);
        if (!c.sliced || !d.sliced) continue;
        ++sliced_tensors;
        // read the full payload; the two cards' fragments must cover every source byte exactly once
        const uint64_t bytes = tensor_payload_bytes(t);
        std::vector<uint8_t> full(bytes);
        std::vector<char> done(bytes, 0);
        if (!platform::pread_all(fd, full.data(), bytes, gg.data_start() + t.offset)) die("pread", t.name);
        bool ok = true;
        for (const CardPlan* cp : {&c, &d})
            for (const dense::tp::Frag& f : cp->frags) {
                if (f.src + f.len > bytes) { ok = false; break; }
                for (uint64_t i = 0; i < f.len && ok; ++i) {
                    if (done[f.src + i]) ok = false;   // two cards claim one byte
                    done[f.src + i] = 1;
                }
                if (!ok) break;
            }
        for (uint64_t i = 0; ok && i < bytes; ++i)
            if (!done[i]) ok = false;   // a byte neither card holds
        if (!ok) check(false, t.name + ": the two cards' fragments miss or double-cover bytes");
    }
    platform::close_fd(fd);
    const int want_slices = g.gdn_layers() * 11 + g.attn_layers() * 7;   // per mixer kind, the slice classes
    check(sliced_tensors == want_slices, std::to_string(sliced_tensors) + " layer tensors sliced, the table holds " + std::to_string(want_slices));
    check(rep0 >= (long long) (want0 >> 20) - 2 && rep0 <= (long long) (want0 >> 20) + 2,
          "rank 0 reported " + std::to_string(rep0) + " MiB, the table says " + std::to_string(want0 >> 20));
    check(rep1 >= (long long) (want1 >> 20) - 2 && rep1 <= (long long) (want1 >> 20) + 2,
          "rank 1 reported " + std::to_string(rep1) + " MiB, the table says " + std::to_string(want1 >> 20));
    check(want1 * 2 > want0 && want1 * 2 < want0 * 5, "rank 1 holds about half of rank 0 minus the head and drafter");
    (void) axis_len;
}
}  // namespace

int main(int argc, char** argv) {
    const char* env = std::getenv("BASALT_QWEN35_PACK");
    const std::string pack = argc > 1 ? argv[1] : env ? env : "";
    int devices = 0;
    if (pack.empty() || cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::printf("skip: no 27B pack (argument or BASALT_QWEN35_PACK) or fewer than two GPUs\n");
        return 77;
    }
    core::ModelFile file;
    std::string err;
    if (!file.open(pack, err) || !file.load_meta(err) || !file.register_gguf_view(err)) die(pack.c_str(), err);
    const std::unique_ptr<model::ModelAdapter> a = models::qwen35::make_adapter();
    try_load(*a, pack, false);
    try_load(*a, pack, true);
    // --tp and a layer cut refuse each other
    model::LoadRequest req;
    req.paths = {pack};
    req.max_context = 8192;
    req.tp = 2;
    req.diag_split_at = 5;
    std::unique_ptr<model::LoadedModel> loaded;
    const model::Result r = a->load(req, loaded);
    check(!r.ok() && r.message.find("do not mix") != std::string::npos, "--tp with --diag-split-at refuses: " + r.message);
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
