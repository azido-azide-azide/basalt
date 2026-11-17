// tests/dense/tp_geometry_test.cpp - the widths each card of a tensor-parallel pair runs, on a 27B-class geometry (24
// query heads over 4 key-value heads of 256, a delta-net of 16 key heads and 48 value heads of 128, 5120 wide, 17408
// intermediate, 64 layers with every fourth one attention): the heads and widths sharing out whole, the pair's widths
// adding back to the model's, each card's delta-net being gdn_head_split.hpp's smaller layer of its key-head group, and
// a card's geometry still being one the shared kernels take.  Host only - no device, no pack.
#include "basalt/dense/tp_geometry.hpp"
#include "tp_test_model.hpp"

#include <cstdio>
#include <string>
#include <vector>

using basalt::dense::Geometry;
using basalt::core::GdnHeadSlice;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// what Geometry::check() asks of the widths and head grouping, without the kernels' own tables (which the device test
// covers): a card's geometry must keep passing these, or a card could not run the shared layer kernels.
bool widths_ok(const Geometry& g) {
    if (g.n_embd % 256 || g.n_ff % 256) return false;
    if (g.n_kv <= 0 || g.n_head % g.n_kv || g.n_head / g.n_kv > 8) return false;
    if (g.head_dim != 256) return false;
    if (g.rope.rot <= 0 || g.rope.rot > g.head_dim || g.rope.rot % 2 || g.rope.rot / 2 > 64) return false;
    return true;
}

}  // namespace

int main() {
    const Geometry g = basalt::dense::tp_test::qwen38_27b();
    std::string err;

    // one card holds the model as it is
    Geometry solo;
    check(basalt::dense::tp_card_geometry(g, 1, 0, solo, err) && solo.n_head == g.n_head && solo.n_kv == g.n_kv &&
              solo.gdn_k_heads == g.gdn_k_heads && solo.n_ff == g.n_ff,
          "a width of one is the model's own geometry");
    err.clear();
    check(!basalt::dense::tp_card_geometry(g, 1, 1, solo, err) && !err.empty(), "a width of one has only card 0");

    // widths that do not share out are refused, not rounded
    for (int w : {3, 5, 6, 7}) {
        err.clear();
        Geometry cg;
        check(!basalt::dense::tp_card_geometry(g, w, 0, cg, err) && !err.empty(),
              "a pair of " + std::to_string(w) + " is refused with a reason (" + err + ")");
    }

    // the pair: each card's widths, and the pair adding back to the model
    Geometry c0, c1;
    check(basalt::dense::tp_card_geometry(g, 2, 0, c0, err) && basalt::dense::tp_card_geometry(g, 2, 1, c1, err),
          "the pair of two is taken");
    check(c0.n_head == 12 && c0.n_kv == 2 && c0.gdn_k_heads == 8 && c0.gdn_v_heads == 24 && c0.n_ff == 8704,
          "card 0 of two: 12 query and 2 key-value heads, 8 key and 24 value delta-net heads, 8704 intermediate");
    using WidthFn = int (Geometry::*)() const;
    const auto sum = [&](WidthFn f) { return (c0.*f)() + (c1.*f)(); };
    check(c0.n_head + c1.n_head == g.n_head && c0.n_kv + c1.n_kv == g.n_kv &&
              c0.gdn_k_heads + c1.gdn_k_heads == g.gdn_k_heads && c0.gdn_v_heads + c1.gdn_v_heads == g.gdn_v_heads &&
              c0.n_ff + c1.n_ff == g.n_ff,
          "the pair's heads and intermediate width add back to the model's");
    check(sum(&Geometry::conv_channels) == g.conv_channels() && sum(&Geometry::value_width) == g.value_width(),
          "the pair's delta-net channels and value width add back to the model's");
    check(c0.n_embd == g.n_embd && c0.head_dim == g.head_dim && c0.gdn_state == g.gdn_state && c0.eps == g.eps &&
              c0.n_vocab == g.n_vocab && c0.n_layers == g.n_layers,
          "the shared widths - n_embd, head width, delta-net head size, epsilon, vocabulary - stay the model's");
    check(c0.n_head % c0.n_kv == 0 && c0.n_head / c0.n_kv == g.n_head / g.n_kv,
          "a card's queries meet their own keys in the same group ratio");
    check(widths_ok(c0) && widths_ok(c1), "each card's geometry is still one the layer kernels take");

    // the key-head group is the packed widths the card's geometry claims, for both cards
    for (int r : {0, 1}) {
        const GdnHeadSlice sl = basalt::dense::tp_gdn_group(g, 2, r);
        const Geometry& cg = r ? c1 : c0;
        check(sl.k0 == r * 8 && sl.nk == 8 && sl.nv() * 2 == g.gdn_v_heads && sl.c() == cg.conv_channels() &&
                  sl.zv() == cg.value_width(),
              "card " + std::to_string(r) + "'s key-head group packs to its own channels and value width");
    }

    // the state each card holds is its own heads' only: the pair adds back to one card's whole state
    const size_t whole = (size_t) g.gdn_state * g.gdn_v_heads * g.gdn_state + (size_t) g.conv_channels() * g.gdn_conv;
    const size_t pair = basalt::dense::tp_gdn_state_floats(g, 2, 0, g.gdn_conv) +
                        basalt::dense::tp_gdn_state_floats(g, 2, 1, g.gdn_conv);
    check(pair == whole && whole / 2 < whole, "the pair's recurrent state and conv history add back to one layer's");
    check(basalt::dense::tp_gdn_state_floats(g, 2, 0, g.gdn_conv) < whole, "and neither card holds the whole state");

    // the cells each card's attention caches are its key-value heads' only
    check((c0.n_kv + c1.n_kv) * c0.head_dim == g.n_kv * g.head_dim && c0.n_kv * c0.head_dim < g.n_kv * g.head_dim,
          "a card's cache row holds its key-value heads, and the pair the model's");

    // the head's rows share out; a vocabulary that does not divide has no shard
    check(basalt::dense::tp_head_rows(g, 2, 0) == 124160 && basalt::dense::tp_head_rows(g, 2, 1) == 124160 &&
              basalt::dense::tp_head_rows(g, 2, 0) + basalt::dense::tp_head_rows(g, 2, 1) == g.n_vocab,
          "the head's 248320 rows split 124160 and 124160");
    check(basalt::dense::tp_head_rows(g, 3, 0) == 0, "a vocabulary that does not divide has no head shard");

    // four cards on the same model: 4 key heads and 12 value heads a card, one key-value head, 4352 intermediate
    Geometry d[4];
    bool ok4 = true;
    err.clear();
    for (int r = 0; r < 4; ++r) ok4 = ok4 && basalt::dense::tp_card_geometry(g, 4, r, d[r], err);
    size_t kv = 0, st = 0;
    for (int r = 0; r < 4; ++r) {
        kv += (size_t) d[r].n_kv;
        st += basalt::dense::tp_gdn_state_floats(g, 4, r, g.gdn_conv);
    }
    check(ok4 && d[0].n_head == 6 && d[0].n_kv == 1 && d[0].gdn_k_heads == 4 && d[0].gdn_v_heads == 12 &&
              d[0].n_ff == 4352 && widths_ok(d[3]) && kv == (size_t) g.n_kv && st == whole,
          "a four-card share keeps every width whole and the state's total the same (" + err + ")");

    std::printf("%s\n", failures ? "FAILED" : "all good");
    return failures ? 1 : 0;
}
