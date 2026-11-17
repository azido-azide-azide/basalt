// tests/dense/tp_test_model.hpp - the geometry the tensor-parallel tests share: Qwen3.8-27B as its pack's metadata gives
// it (24 query heads over 4 key-value heads of 256, a delta-net of 16 key heads and 48 value heads of 128, 5120 wide,
// 17408 intermediate, 64 layers with every fourth one attention, 262144 of context).  One copy on purpose: the pair's
// widths, the state it splits and the buffers it sizes are all checked against THESE numbers, and a hand-copied variant
// would let a test pass on a model nobody ships.
#pragma once

#include "basalt/dense/model.hpp"

#include <vector>

namespace basalt::dense::tp_test {

inline Geometry qwen38_27b() {
    Geometry g;
    g.n_layers = 64;
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
    g.rope.rot = 64;
    g.rope.theta = 1.0e7f;
    g.eps = 1e-6f;
    g.kinds.assign((size_t) g.n_layers, basalt::dense::Mixer::Gdn);
    for (int l = 3; l < g.n_layers; l += 4) g.kinds[(size_t) l] = basalt::dense::Mixer::Attention;
    g.context_length = 262144;
    return g;
}

}  // namespace basalt::dense::tp_test
