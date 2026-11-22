# Dense tensor-parallel port onto `main` — scope + checklist

Written 2026-11-22. Repo `/home/hermes/work/basalt-tp-1117`, branch **`dense-tp-main-1122`**
(created from `origin/main` = `ce0156a0`, the fork's main on GitHub — upstream `1c97f52` + the
fork stack through PR #34, live in CT132 `/root/basalt`).

Source of the port: branch `tp-10-fn-outproj-table-fix` (`b936f022`) = old main `2c339435` + 86
commits. 60 of those are FN-MoE (qwen4exp) and are **out of scope**; the dense set is 26 commits
touching only `src/dense`, `src/models/qwen35`, `src/core/tp_mailbox`, `include/basalt/{core,dense,kernels}`,
`src/kernels/cuda/{gdn_prompt,dense_ops,elementwise}`, `src/program`, `CMakeLists.txt`, tests.

---

## 1. What the dense TP path is (so the port has a target)

`--tp 2` builds two half-width models; both run **every** layer of every window/chunk. Each
column-parallel product leaves its card's FP32 partial in scratch and the two partials + the BF16
residual meet over a 2-slot mailbox (`core::TpMailbox::exchange_add_bf16`, one rounding). Rank 0
owns the head, sampler, drafter and its KV; rank 1 keeps layers' cells + recurrent state. The
drivers reset per-rank exchange counters, queue rank 1 first, then refuse unequal/odd counts —
lockstep is a hard invariant.

Measured on 2× 5060 Ti (27B, spec-off): decode ≈ solo (−2%), prefill ≈ 1.8× solo. This is the
path to preserve; solo and the layer split must stay bit-identical.

## 2. Port method, file by file

### 2a. Verbatim (base is byte-identical old main → `origin/main`, so `tp-10`'s file *is* the port)

| File | Why verbatim is exact |
|---|---|
| `src/dense/forward.cpp` | md5-identical between `2c339435` and `ce0156a0`; every commit touching it is dense-only |
| `include/basalt/kernels/gdn_prompt.hpp` | md5-identical |
| `src/kernels/cuda/gdn_prompt.cu` | md5-identical |
| `src/models/qwen35/meta.hpp` | md5-identical (not touched by the port) |

### 2b. New files (no base on main) — copy from `tp-10`

`include/basalt/core/tp_mailbox.hpp`, `src/core/tp_mailbox.cu`, `tests/core/tp_mailbox_test.cpp`,
`include/basalt/dense/tp.hpp`, `src/dense/tp.cpp`, `tests/dense/tp_plan_test.cpp` (new dir),
`tests/kernels/gdn_prompt_tp_parity.cpp`, `tests/models/qwen35/tp_shard_load_test.cpp`,
`tests/models/qwen35/tp_forward_test.cpp`.

### 2c. One-line / tiny edits (clean on main)

| File | Edit |
|---|---|
| `include/basalt/model/adapter.hpp` | `LoadRequest::tp` (+1) |
| `include/basalt/program/options.hpp` | `CommonOptions::tp` (+1) |
| `src/program/cli/arguments.cpp` | `else if (a == "--tp") o.tp = …` (+1, context identical) |
| `src/program/cli/help.cpp` | the two `--tp` help lines (context identical; upstream changed other lines nearby) |
| `src/program/generate/adapter.cpp` | `lreq.tp = o.tp;` (+1) |
| `include/basalt/dense/model.hpp` | `attach_tp` declaration (anchors on `attach_split`, present) |
| `src/dense/model.cpp` | 2 × `gdn_work_bytes(T, g.gdn_k_heads, g.gdn_v_heads)` |

### 2d. Real merges (upstream rewrote the same region)

| File | Upstream change | Port work |
|---|---|---|
| `src/models/qwen35/adapter.cpp` | `load()` → `plan()`/`attach()`; prompt line → prompt split (`PromptStage`, `plan/attach_prompt_split`); `LoadPlan` queued reads | the big one — see §3 |
| `src/dense/session.cpp` | `draft_end(…, const DraftNeed&)`; `prompt_split` rename; logit-bias memcpy on the stream | head gate `!m.tp_rank`, `tp_session`, 4 × `hang_reported` checks, `peer_model()/peer_device()` |
| `src/dense/internal.hpp` | `prompt_split` rename, `drafter_steps` | `TpHand`, `Model::tp`/`tp_rank`/`tp_own`, `tp_session` decl, `chunk_layers(…, scan_l1)` |
| `src/dense/stage.cpp` | prompt-line → prompt-split rename | append `TpHand::~TpHand` / `attach_tp` / `tp_session` after `split_session` |
| `src/kernels/cuda/dense_ops.cu` | +136/−3 (bf16_rows_dot tiles, f32_rows_dot, router) | `gdn_gating_proj` NF as a template parameter so a 24-value-head card runs (block content identical, new line numbers) |
| `CMakeLists.txt` | +253/−44 | `src/dense/tp.cpp` into `dense`; `src/core/tp_mailbox.cu` into `core`; 4 test targets; `tests/dense/` |

### 2e. Move (shared infra, currently under the model)

`include/basalt/models/qwen4exp/decode/gdn_head_split.hpp` → `include/basalt/core/gdn_head_split.hpp`.
`dense/tp.hpp` needs `core::GdnHeadSlice`/`gdn_qkv_runs`/`gdn_vhead_runs`; shared code may not include
a model header (`tools/check_model_layering.sh`). Update the 8 include lines in
`src/models/qwen4exp/decode/verify/*.cpp` + `tests/models/qwen4exp/kernels/gdn_head_split_parity.cpp`.
Take `origin/main`'s content (it already has every function dense needs); `tp-10`'s extra
`gdn_local_head`/`gdn_place_groups`/`gdn_grouped_src_col` are FN-only — **skip them**.

### 2f. Explicitly NOT ported

- All `src/models/qwen4exp/**` TP work (`fn_tp_pair`, `tp_window`, `tp_exchange`, `gdn_split`,
  `plan/tp_rank`, `load/*`, `serve/*`), `tests/models/qwen4exp/tp_*`.
- `src/core/weights.{hpp,cpp}` `RowSlices` work (commits `35d850c5`, `1734d2ca`, `6c7d83a0`,
  `781370bc`) and `tests/core/row_slice_sizing_test.cpp`. **Verified: the dense stack never
  references `WeightTable`/`RowSlices`** — the dense load slices with its own
  `dense::tp::payload_fragments` + `pread`, so none of that is on the dense path.
- `src/kernels/cuda/elementwise.{cu,hpp}` `copy_to_mapped_wt`/`copy_from_mapped_cv`/`diag_*` —
  FN hand-off + probe instrumentation only; the dense pair uses `exchange_add_bf16`.
- `tp-dense-1116`'s planner pricing (`MgModel::tp_built`, `SplitInputs::tensor_parallel`,
  `split_plan` TP fit) — a separate branch that is not in `tp-10`. Optional follow-up.
- `gpu_select`/`cli/gpu` role naming (`PromptLine` → `PromptSplit`): TP reuses `peer_devices`,
  so no new role is needed.

## 3. The one real design decision — the sliced reads

`tp-10` reads a card's slices with a private fd + `pread` + staging at load time. `main`'s start
is `plan()` → `place_start()` (one `LoadPlan` pass) → `attach()`. A slice is many contiguous
spans, and `core::LoadItem` is one contiguous span with a conversion — but every dense fragment is
`Conv::Copy`, so **one `LoadItem` per `dense::tp::Frag`** expresses a card's tensor exactly, and
the new loader keeps its threading, its progress bar and its staging.

Plan: keep `tp_slice_for` / `payload_fragments` / `shard_bytes` / `tp_slice_bytes` unchanged; emit
`LoadItem{file, off = data_start + t.offset + f.src, bytes = f.len, dev = carved + f.dst,
conv = Copy}` per fragment into `plan_` instead of `copy_frags`. The block carve (`reserve_block_tp`,
`take`) is unchanged, so the byte accounting the tests check stays identical. Fallback if the item
count is a problem: keep `tp-10`'s private `pread` path inside `plan()`.

Consequences of `main`'s phase split:
- rank 0's model builds in `plan()` (as it does today); **rank 1's model and `attach_tp` move to
  `attach()`** — its weights have landed by then. `dense::make_model` defers all weight-touching
  work to `attach_drafter`, so building rank 1 there is safe.
- the rank-1 headroom gate (`shared_bytes` + `session_bytes` of the card geometry + block drafter
  on rank 0) moves to `attach()` too, where the card's free VRAM is real.
- `--tp` + `--diag-split-at` still refuse each other; `--tp` + MTP drafter and `--tp` + `--vision`
  still refuse (guards carried over).

## 4. Risks

- **Lockstep.** Every body must queue the same even number of exchanges. The drivers already
  refuse unequal/odd `epi`; port that code unchanged.
- **Graph capture.** Both cards' graphs are captured before either launches. Carried in
  `forward.cpp` verbatim.
- **Layering test.** `model_layering` runs in CMake (`ctest`) and in CI; the `gdn_head_split` move
  and the new `core`/`dense` includes must keep it green.
- **Upstream drift in `qwen35/adapter.cpp`.** The `plan`/`attach` split, `PromptStage` and
  `LoadPlan` are all new since the TP branch's base; this is where a silent behavioural mismatch
  would hide. Gate with the shard-load test's byte accounting + a solo/pair A/B.
- **Solo regression.** Every hunk is guarded on `m.tp`; solo and the layer split must stay
  bit-identical. Gate: solo output before/after on the same pack.

## 5. Checklist

### Phase 0 — branch
- [x] Branch `dense-tp-main-1122` created from `origin/main` (`ce0156a0`); working tree clean.
      Confirmed against GitHub: `refs/heads/main` = `ce0156a0` (README: synced to upstream
      `1c97f52`, past 0.5.6). The local `merge-1121` scratch (`d49261fe`) is *not* main.

### Phase 1 — new files & moves
- [ ] `src/core/tp_mailbox.{hpp,cu}` + `tests/core/tp_mailbox_test.cpp` added.
- [ ] `include/basalt/dense/tp.hpp` + `src/dense/tp.cpp` + `tests/dense/tp_plan_test.cpp` added.
- [ ] `tests/kernels/gdn_prompt_tp_parity.cpp` added.
- [ ] `gdn_head_split.hpp` moved to `include/basalt/core/`; 8 include lines updated; header comment
      path fixed; `model_layering` green.
- [ ] `tests/models/qwen35/tp_shard_load_test.cpp` + `tp_forward_test.cpp` added.

### Phase 2 — verbatim dense kernel/forward files
- [ ] `src/dense/forward.cpp` taken from `tp-10` (md5 of base verified identical first).
- [ ] `include/basalt/kernels/gdn_prompt.hpp` + `src/kernels/cuda/gdn_prompt.cu` taken from `tp-10`.
- [ ] `src/dense/model.cpp`: 2 `gdn_work_bytes` head-count call sites.
- [ ] `src/kernels/cuda/dense_ops.cu`: `gdn_gating_proj` NF template hunk (24-head card).

### Phase 3 — dense stack merges
- [ ] `include/basalt/dense/model.hpp`: `attach_tp` decl.
- [ ] `src/dense/internal.hpp`: `TpHand`, `Model::tp*`, `tp_session`, `chunk_layers(scan_l1)`.
- [ ] `src/dense/stage.cpp`: `TpHand::~TpHand` / `attach_tp` / `tp_session` appended.
- [ ] `src/dense/session.cpp`: head gate, `tp_session`, 4 `hang_reported`, `peer_model()/peer_device()`.

### Phase 4 — the adapter (the big one)
- [ ] `plan_load`: TP plan check, `on_card_tp`, `reserve_block_tp` both ranks, per-fragment
      `LoadItem`s, rank-0 model on the card geometry.
- [ ] `attach`: rank-1 model + `attach_tp`, rank-0/rank-1 headroom gates, sessions.
- [ ] `linear`/`floats`/`bf16_pair` take the `plan_` branch → `upload_slice`/`bf16_pair_slice`.
- [ ] `--tp` in `parse_flag` + `finish_flags` guards (card count, diag-split, MTP, vision).
- [ ] `lreq.tp` wired in `src/program/generate/adapter.cpp`.

### Phase 5 — build
- [x] `CMakeLists.txt`: `src/dense/tp.cpp` → `dense`; `src/core/tp_mailbox.cu` → `core`;
      4 test targets; `tests/dense/`.
- [x] Build on CT132 (`/root/basalt-tpmain`): `cmake --build build --target basalt -j8` — clean.
- [x] `model_layering` + host tests green (see Phase 6).

### Phase 6 — tests (all eight green; the two 27B ones need the cards free)
- [x] `dense_tp_plan_test` — Passed 2.18 s.
- [x] `qwen35_split_plan_test`, `gdn_head_split_parity` — Passed.
- [x] `tp_mailbox_test` — Passed 41.7 s.
- [x] `gdn_prompt_tp_parity` — Passed.
- [x] `dense_ops_parity` — **PASS** on quiet cards (`gdn_conv` 396.7 µs, 423 GB/s,
      `candidate_path` 0/7 picks differ). It OOMs (`gdn_conv: out of memory`) while the FN
      engine is resident — the test holds ~173 MB and the engine leaves ~200 MB; **main's own
      binary fails identically**, and the port touches no allocation in that file.
- [x] `qwen35_tp_shard_load_test` (27B + 2 GPUs) — **all ok**: both card models built and paired,
      rank 1 holds 7050 MiB (table says 6553, no drafter bytes — the drafter is rank 0's alone),
      rank 0 carries the block drafter, and `--tp` + `--diag-split-at` refuses.
- [x] `qwen35_tp_forward_test` (27B + 2 GPUs) — **all ok**: chunk prefill and six window decodes
      ran on the pair.

### Phase 7 — live gates (2× 5060 Ti, 27B)
- [x] **Solo bitwise vs main's binary**: same pack/config, 5 greedy prompts @256 tok — **5/5
      byte-identical** text, identical `finish_reason`. (`bw-port.jsonl` vs `bw-base.jsonl`.)
- [x] Pair vs solo: spec-off decode **1.72×** (28.5 → 49.1 tok/s), prefill **1.20×**
      (1342 → 1607 tok/s). Spec-8: decode **1.50×** (92.2 → 138.2), prefill **1.21×**
      (1301 → 1579). 8 prompts × 3 passes @512 and × 2 @256.
- [x] Greedy quality sweep: coherent English on all 8 prompts, `finish_reason=stop` everywhere
      except the one prompt that ran to the token cap (`length`) on both arms alike.
- [x] `--tp 2` + `--diag-split-at` refuses (in the shard-load test); `--tp` + MTP and `--tp` +
      `--vision` guards carried over.

### Phase 8 — publish
- [x] Push `dense-tp-main-1122` (`7c4b1d7c`).
- [ ] Open PR against `main` — **awaiting Leon's explicit go for the PR act**.

---

## 6. Gauntlet result (the TP-over-TS measurement)

2× RTX 5060 Ti, Qwen3.8-27B-QUASAR-NVFP4, greedy `temp=0`, Berlin-tested 2026-11-22.
TS = the layer split (what `--diag-split-at`/auto-plan picks); TP = `--tp 2` (the port).

| arm | prefill tok/s | decode tok/s |
|---|---|---|
| ts (layer split), spec 8 | 1300.6 | 92.2 |
| **tp, spec 8** | **1578.9 (1.21×)** | **138.2 (1.50×)** |
| ts (layer split), spec 0 | 1342.0 | 28.5 |
| **tp, spec 0** | **1607.1 (1.20×)** | **49.1 (1.72×)** |

Medians over 8 prompts; draft acceptance 55.2% (ts) vs 52.2% (tp). Per-prompt decode ratio
1.31–1.75×, prefill 1.05–1.23×. Chart: `gauntlet/gauntlet.png`.
