// tests/dense/tp_sizing_test.cpp - what the tensor-parallel pair's card geometry does to the buffers a card reserves for
// ITSELF: the per-session layer bytes (key-value cells, delta-net state and conv history) and the shared decode/prompt
// scratch.  Pure arithmetic - it asks the sizing functions, not a device - but it lives next to libdense because that is
// where those formulas are, so the build registers it with the GPU tests and a machine without them skips it.
#include "basalt/dense/tp_geometry.hpp"
#include "tp_test_model.hpp"

#include <cstdio>
#include <string>

using basalt::dense::Geometry;
using basalt::dense::Mixer;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

}  // namespace

int main() {
    const Geometry g = basalt::dense::tp_test::qwen38_27b();
    Geometry c[2];
    std::string err;
    for (int r = 0; r < 2; ++r) {
        if (!basalt::dense::tp_card_geometry(g, 2, r, c[r], err)) {
            std::printf("FAIL card %d: %s\n", r, err.c_str());
            return 1;
        }
    }

    // The state one card's delta-net layer holds, and the conservation the slicing owes it: the two cards' recurrent state
    // and conv history together are exactly what one card's whole layer held before the pair existed (hist 3: the sizing
    // path's conv history depth, so the numbers below are the ones a session really asks for).
    const size_t full_state = (size_t) g.gdn_state * g.gdn_v_heads * g.gdn_state + (size_t) g.conv_channels() * 3;
    check(basalt::dense::tp_gdn_state_floats(g, 1, 0, 3) == full_state, "one card of one card is the whole layer's state");
    check(basalt::dense::tp_gdn_state_floats(g, 2, 0, 3) + basalt::dense::tp_gdn_state_floats(g, 2, 1, 3) == full_state,
          "the pair's delta-net state and conv history add back to the layer's");

    // Two kinds of bytes: the ones a pair splits, and the ones it pays twice.  Everything head-shaped - a session's
    // key-value cells, a delta-net layer's state and conv history - halves exactly.  Three things do not: each attention
    // layer's fixed table, the session's fixed base, and the logits window, which the card geometry keeps at the model's
    // full vocabulary ON PURPOSE (card 0's logits hold every row once the pair gathers them, and a card that sized them
    // narrow would overflow the gather).  Rather than restate those constants, the test asks the sizing for them: an
    // attention layer at zero context is nothing but its fixed table, and doubling the vocabulary is nothing but the
    // window that follows it.
    int attn_layer = -1, n_attn = 0;
    for (int il = 0; il < g.n_layers; ++il)
        if (g.kinds[(size_t) il] == Mixer::Attention) {
            if (attn_layer < 0) attn_layer = il;
            ++n_attn;
        }
    const uint64_t attn_fixed = basalt::dense::session_layer_bytes(g, attn_layer, /*max_context=*/0, /*kv_int8=*/true, 0);
    Geometry wide = g;
    wide.n_vocab *= 2;
    const uint64_t vocab_window =
        basalt::dense::session_bytes(wide, 131072, false, true, 131072) - basalt::dense::session_bytes(g, 131072, false, true, 131072);
    const uint64_t session_base = 64 * 256;   // session_bytes' fixed opening, in bytes (src/dense/model.cpp)
    std::printf("#  unsplit: %llu B per attention layer x %d, session base %llu B, logits window %llu MiB\n",
                (unsigned long long) attn_fixed, n_attn, (unsigned long long) session_base,
                (unsigned long long) (vocab_window >> 20));

    const int64_t ctx = 131072;
    for (int il = 0; il < g.n_layers; ++il) {
        const bool attn = g.kinds[(size_t) il] == Mixer::Attention;
        const uint64_t all = basalt::dense::session_layer_bytes(g, il, ctx, /*kv_int8=*/true, ctx);
        const uint64_t half = basalt::dense::session_layer_bytes(c[0], il, ctx, /*kv_int8=*/true, ctx);
        if (il < 4)   // the pack's first three layers are delta-net, the fourth attention (interval 4)
            std::printf("#  layer %2d (%s): whole %llu B, one card %llu B, the pair %llu B\n", il, attn ? "attn" : "gdn ",
                        (unsigned long long) all, (unsigned long long) half, (unsigned long long) (2 * half));
        check(2 * half == all + (attn ? attn_fixed : 0),
              std::string("layer ") + std::to_string(il) + (attn ? " (attn): its cells halve and its fixed "
              + std::to_string(attn_fixed) + " B table is paid twice"
                                                          : " (gdn): its state and history halve with no fixed part"));
    }

    // The scratch a card reserves for itself shrinks, but by less than half: same three unsplit terms, and the scratch
    // holds the logits buffer and the logprob scratch at the model's vocabulary.
    const uint64_t s_all = basalt::dense::shared_bytes(g, ctx, /*kv_int8=*/true, ctx);
    const uint64_t s_card = basalt::dense::shared_bytes(c[0], ctx, /*kv_int8=*/true, ctx);
    std::printf("#  shared decode+prompt scratch: whole model %.2f GiB, one card %.2f GiB (%.0f%% of it)\n",
                (double) s_all / 1073741824.0, (double) s_card / 1073741824.0, 100.0 * (double) s_card / (double) s_all);
    check(s_card < s_all, "a card's scratch is smaller than the whole model's");
    check(2 * s_card > s_all, "and shrinks by less than half: the vocabulary-wide buffers stay whole");

    // A session's bytes on a card: every splitting term halves, and the pair's two sessions together are the model's
    // session plus the terms it pays twice - named above, to the byte.
    const uint64_t sess_all = basalt::dense::session_bytes(g, ctx, /*mtp=*/false, /*kv_int8=*/true, ctx);
    const uint64_t sess_card = basalt::dense::session_bytes(c[0], ctx, /*mtp=*/false, /*kv_int8=*/true, ctx);
    std::printf("#  one session at %lld: whole model %.2f GiB, one card %.2f GiB; the pair pays %lld KiB twice\n",
                (long long) ctx, (double) sess_all / 1073741824.0, (double) sess_card / 1073741824.0,
                (long long) ((2 * sess_card - sess_all) >> 10));
    check(2 * sess_card == sess_all + (uint64_t) n_attn * attn_fixed + session_base + vocab_window,
          "a card's session is exactly half the model's, and the two cards together pay its fixed tables and its "
          "vocabulary-wide logits window twice");

    std::printf("%s\n", failures ? "FAILED" : "all good");
    return failures ? 1 : 0;
}
