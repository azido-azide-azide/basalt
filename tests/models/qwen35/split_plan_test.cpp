// tests/models/qwen35/split_plan_test.cpp - dense::plan_split on Qwen3.8-27B's bytes: the main card alone when the model
// fits it, else the cut that puts the most layers on the faster of two unlike cards or halves the layers' bytes between alike
// ones, an error when neither fits, and a forced cut taken as given when it fits.  The bytes are what the 27B pack (NVFP4 layers,
// FP8 head, block drafter) and dense::split_inputs give for each setting; the free memory is what the cards report at start.
#include "basalt/dense/split_plan.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using basalt::dense::SplitCard;
using basalt::dense::SplitInputs;
using basalt::dense::SplitPlan;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

constexpr uint64_t kGdnLayer = 216476160, kAttnLayer = 209432320;    // weights: a GDN layer, an attention layer (256-byte aligned)
constexpr uint64_t kGdnSlot = 7854080;                               // a slot's recurrent state in a GDN layer
constexpr uint64_t kMainFixed = 5918443012, kMainFixed8K = 5582898692;   // head, drafter, scratch, a slot's logits (spec 16)
constexpr uint64_t kSlotRest = 20000000;                              // another slot's logits and drafter ring
constexpr uint64_t kPeerFixed = 2165045508;                           // the second card's scratch and landing buffers
constexpr uint64_t kMargin = 512ull << 20;
constexpr uint64_t k5090 = 32622837760, k16GB = 15979839488;          // free at start: RTX 5090, a 16 GB card

SplitCard card(uint64_t free, double gbps) { return {free - kMargin, gbps}; }

// every fourth layer attention; `attn_slot`: a slot's cells in an attention layer at the setting's context and KV format
SplitInputs inputs(uint64_t attn_slot, int slots, uint64_t main_fixed, SplitCard main, SplitCard peer, int forced = 0) {
    SplitInputs in;
    for (int l = 0; l < 64; ++l) {
        const bool attn = l % 4 == 3;
        in.layer.push_back(attn ? kAttnLayer : kGdnLayer);
        in.per_slot.push_back(attn ? attn_slot : kGdnSlot);
    }
    in.slots = slots;
    in.main_fixed = main_fixed + (uint64_t) (slots - 1) * kSlotRest;
    in.peer_fixed = kPeerFixed;
    in.main = main;
    in.peer = peer;
    in.forced = forced;
    return in;
}

void expect(const char* what, const SplitInputs& in, int cut) {
    const SplitPlan p = basalt::dense::plan_split(in);
    const bool fits = p.error.empty() && p.main_bytes <= in.main.free && p.peer_bytes <= in.peer.free;
    check(p.cut == cut && fits, std::string(what) + ": cut " + std::to_string(p.cut) + " (want " + std::to_string(cut) + ")" +
                                    (p.error.empty() ? "" : ", " + p.error));
}

void expect_error(const char* what, const SplitInputs& in) {
    const SplitPlan p = basalt::dense::plan_split(in);
    check(!p.error.empty() && p.cut == 0, std::string(what) + ": refused (" + p.error + ")");
}

}  // namespace

int main() {
    constexpr uint64_t fp16_8k = 33555456, fp16_262k = 1073742848, int8_128k = 276825088, int8_262k = 553649152;
    const SplitCard r5090 = card(k5090, 1792), r5060 = card(k16GB, 448), r5070 = card(k16GB, 896);
    // RTX 5090 + RTX 5060 Ti: unlike cards, the most layers on the 5090
    expect("32+16, 8K fp16 (fits the main card)", inputs(fp16_8k, 1, kMainFixed8K, r5090, r5060), 0);
    expect("32+16, 262K fp16", inputs(fp16_262k, 1, kMainFixed, r5090, r5060), 55);
    expect("32+16, 262K int8, two slots", inputs(int8_262k, 2, kMainFixed, r5090, r5060), 51);
    {   // --parallel 2 through the server (every kernel loaded at start: less free); these cut the model, then ran short at load
        // while each weight was an allocation of its own, rounded up to the allocator's 2 MiB pages
        const SplitCard m5090{32515883008ull, 1792}, m5060{15872884736ull, 448};   // free past the margin, as planned
        const auto two = [&](uint64_t attn_slot) {
            SplitInputs in = inputs(attn_slot, 2, kMainFixed, m5090, m5060);
            in.main_fixed = 5981882116ull;   // head, drafter, scratch, both slots' logits and drafter rings
            return in;
        };
        expect("32+16, 262K int8, two slots, served", two(553649152), 53);
        expect("32+16, 229K int8, two slots, served", two(484443136), 57);
        expect("32+16, 196K int8, two slots, served", two(415237120), 62);
    }
    // a slower main card than the second (listed first): the most layers on the second card
    {
        const SplitPlan p = basalt::dense::plan_split(inputs(fp16_262k, 1, kMainFixed, card(k5090, 448), card(k5090, 1792)));
        check(p.error.empty() && p.cut > 0 && p.cut < 20, "a slower main card keeps the fewest layers that fit it: cut " + std::to_string(p.cut));
    }
    // two RTX 5090s: alike, the layers' bytes halved
    expect("32+32, 262K fp16, two slots", inputs(fp16_262k, 2, kMainFixed, r5090, r5090), 32);
    // two 16 GB cards: the weights alone do not fit one
    expect("16+16, 8K fp16", inputs(fp16_8k, 1, kMainFixed8K, r5070, r5070), 32);
    expect("two RTX 5070 Ti, 128K int8, spec 16 (#376)", inputs(int8_128k, 1, kMainFixed, r5070, r5070), 32);
    expect_error("16+16, 262K fp16 (fits neither one card nor a split)", inputs(fp16_262k, 1, kMainFixed, r5070, r5070));
    // width-2 tensor parallel: half of every layer's bytes and half of every slot's states on each card, plus fixed parts
    {
        SplitInputs fit = inputs(int8_128k, 1, kMainFixed, r5070, r5070);
        fit.tensor_parallel = true;
        const SplitPlan p = basalt::dense::plan_split(fit);
        check(p.tp_fits == (p.tp_main <= r5070.free && p.tp_peer <= r5070.free) && p.cut == 0 &&
              p.tp_main > 0 && p.tp_peer > 0,
              "16+16, 128K int8 tensor-parallel fit: " + std::to_string(p.tp_main >> 20) + " + " +
              std::to_string(p.tp_peer >> 20) + " MiB");
        SplitInputs nofit = inputs(fp16_262k, 1, kMainFixed, r5070, r5070);
        nofit.tensor_parallel = true;
        const SplitPlan q = basalt::dense::plan_split(nofit);
        check(!q.tp_fits && q.error.find("tensor parallel") != std::string::npos,
              "262K fp16 cells do not fit the pair even split across every layer");
    }
    // the head's vocabulary is the fixed part the pair shares out: every card runs half the rows, so half those bytes
    // move to the second card and the pair's total does not change (the 27B pack's FP8 head and its per-row scales)
    {
        constexpr uint64_t kHead = 1271398400 + 993280;   // output.weight (F8_E4M3) + output.scale (F32 per row)
        SplitInputs fit = inputs(int8_128k, 1, kMainFixed, r5070, r5070);
        fit.tensor_parallel = true;
        const SplitPlan whole = basalt::dense::plan_split(fit);
        fit.main_fixed_splittable = kHead;
        const SplitPlan half = basalt::dense::plan_split(fit);
        check(half.tp_main == whole.tp_main - kHead / 2 && half.tp_peer == whole.tp_peer + kHead / 2 &&
                  half.tp_main + half.tp_peer == whole.tp_main + whole.tp_peer,
              "the head shares out between the pair: " + std::to_string(half.tp_main >> 20) + " + " +
              std::to_string(half.tp_peer >> 20) + " MiB (was " + std::to_string(whole.tp_main >> 20) + " + " +
              std::to_string(whole.tp_peer >> 20) + ")");
        // a main card that bare that the head is what tips it over: the pack's own fit, minus the head's half
        const SplitCard tight = card(k16GB - kHead / 2, 896);
        SplitInputs tip = inputs(int8_128k, 1, kMainFixed, tight, r5070);
        tip.tensor_parallel = true;
        check(!basalt::dense::plan_split(tip).tp_fits, "the head whole on the main card tips the tighter pair over");
        tip.main_fixed_splittable = kHead;
        check(basalt::dense::plan_split(tip).tp_fits, "half the head is what fits the tighter pair");
        SplitInputs over = tip;
        over.main_fixed_splittable = kMainFixed * 2;   // more than the main card holds at all
        const SplitPlan o = basalt::dense::plan_split(over);
        check(o.tp_main == whole.tp_main - kMainFixed / 2, "bytes beyond the main card's own cannot move to the second");
    }
    // the diagnostic cut
    expect("32+16, 8K, forced at 40", inputs(fp16_8k, 1, kMainFixed8K, r5090, r5060, 40), 40);
    expect("32+16, 8K, forced at 8", inputs(fp16_8k, 1, kMainFixed8K, r5090, r5060, 8), 8);
    expect_error("32+16, 8K, forced at 4 (the second card cannot hold the rest)", inputs(fp16_8k, 1, kMainFixed8K, r5090, r5060, 4));
    expect_error("forced at 64 (no layer left for the second card)", inputs(fp16_8k, 1, kMainFixed8K, r5090, r5060, 64));
    check(basalt::dense::split_cards_alike({0, 896}, {0, 960}) && !basalt::dense::split_cards_alike({0, 1792}, {0, 448}),
          "cards within 10% of each other's bandwidth are alike");
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
