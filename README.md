<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/basalt-lockup-dark.png"><img src="assets/basalt-lockup-light.png" alt="Basalt" width="480"></picture></p>

## This fork

Upstream, plus the changes one deployment runs on top of it:

- **A shared arena on hugetlbfs** (`src/platform/memory.cpp`): `--shared-expert-arena` takes a file on a
  huge-page mount, not only `/dev/shm`. The file is truncated to a multiple of the huge page and its header
  is reached through the mapping, since hugetlbfs accepts neither an unrounded `ftruncate` nor `read`/`write`;
  a size mismatch is then told by the header, naming the size it wanted.
- **`--load-threads N`** (`src/models/qwen4exp/flags.cpp`, `src/models/qwen4exp/load/sources.cpp`,
  `docs/configuration/index.md`): the thread count of the start-up arena read. Upstream hardcodes 6 at the
  call site; this fork runs 16. A flag and a config entry, so it follows `CONTRIBUTING.md` (no
  environment-variable knobs).
- **Console figures** (`serve/service.py`, `serve/handler.py`, `serve/util/logbuf.py`, `serve/web/`): a P85 of
  the decode and prefill rates stands beside the Best (the hero figures' tooltips and the Requests panel),
  over best's own gates, and the log level filter counts every level over the server's lifetime and reads a
  level's newest lines through `/api/logs?level=`, so boot-time lines show up although no client kept them.
- **`dash/`**: a second, always-up observer with its own small server, so the panels and the request history
  survive an engine restart, with an optional `--api-key` gate on its data endpoints.
- **`"vision-threads": N`** (`serve/config.py`, `serve/engine/vision.py`, `serve/server.py`): a model key that
  passes `--threads N` to the `basalt-vision` sidecar, bounding the CPU burst of a `vision: "cpu"` encode.
- **Chat slot affinity** (`serve/handler.py`, `serve/engine/process.py`): a multi-turn chat that names no
  session identifies itself by its first user message, so a client that resends the whole history keeps one
  slot even when a turn's re-rendered prompt matches none of the held K/V.

Nothing else is changed.

Synced to upstream main at `db8c8948` (0.5.7, 2026-11-24). The BF16 PLE rows this fork pioneered shipped upstream in
0.3.1 ([#185](https://github.com/jesdga95/basalt/pull/185)), and this rig's community runs now live upstream in
`docs/community/benchmarks.json`, so neither is carried here any more.

Basalt is a local inference engine for large language models: **Qwen3.8-Flash-Next**, **DeepSeek-V4.1-Flash**, **Qwen3.8-27B**, **Hy4-flash** and **Inkling 2 Mini**. It runs
on an NVIDIA Blackwell GPU (sm_120) on Linux and Windows x64, with what does not fit in VRAM (the experts of a mixture-of-experts model)
in system RAM, and optionally on more GPUs. The server speaks the OpenAI and Anthropic APIs. Website: [basalt.jgulfo.com](https://basalt.jgulfo.com).

<p align="center"><img src="docs/img/frontend.gif" alt="The Basalt web UI while a reply streams" width="720"></p>

**Help wanted:** spare Blackwell compute for testing? See [Contributing](CONTRIBUTING.md#other-ways-to-help).

<p align="center">
  <a href="https://github.com/sponsors/jesdga95"><img src="assets/donate/github.svg" height="44" alt="Sponsor on GitHub"></a>
  <a href="https://ko-fi.com/jesdga"><img src="assets/donate/kofi.svg" height="44" alt="Support on Ko-fi"></a>
  <a href="https://paypal.me/jesdga"><img src="assets/donate/paypal.svg" height="44" alt="Donate on PayPal"></a>
</p>

## Requirements

- Linux x86-64 with AVX2, or Windows 10 1903+ x64 ([platforms](docs/platforms.md)).
- An NVIDIA Blackwell GPU (compute capability 12.0) with CUDA 13.
- CMake 3.24+, GCC 13+ (Linux), Python 3.10+, and enough system RAM for the experts that do not fit in VRAM.

## Quick start

Prebuilt Linux and Windows archives are attached to each [release](https://github.com/jesdga95/basalt/releases) ([what they need](docs/platforms.md#release-archives)); otherwise build from source:

```sh
git clone https://github.com/jesdga95/basalt && cd basalt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j                  # bin/basalt, bin/basalt-bench, bin/basalt-probe, bin/basalt-vision
                                        # Windows: add -G Ninja to the first line, from an x64 VS 2022 prompt ([platforms](docs/platforms.md#building-on-windows))
                                        # one model only: -DBASALT_MODELS=qwen4exp, deepseek41, qwen35, hy4flash or inkling2 ([building](docs/building.md#building-only-some-models))
pip install -r requirements.txt
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-NVFP4-DQ6_K.basalt --local-dir models
```

Put the settings in `basalt_config.json`:

```json
{
  "version": 2,
  "models": [{ "model": "models/Qwen3.8-FN-NVFP4-DQ6_K.basalt", "gpu": "0,1", "gpu-mode": "peer", "kv": "int8" }],
  "engine": { "resident-experts": true },
  "server": { "port": 8080 }
}
```

```sh
bin/basalt serve --config basalt_config.json
```

The server listens on 127.0.0.1. Every setting: [configuration](docs/configuration/index.md). More examples:
[example configurations](docs/configuration/examples.md).

## Models

| | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash | Qwen3.8-27B | Hy4-flash | Inkling 2 Mini |
|---|---|---|---|---|---|
| Packs | default `Qwen3.8-FN-NVFP4-DQ6_K`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) | default `DeepSeek-V4.1-Flash-1017-UD-IQ1_M`, [Hugging Face](https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt) | `Qwen3.8-27B-QUASAR-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-27B-Basalt) | `Hy4-flash-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Hy4-flash-Basalt) | `Inkling-2-Mini-NVFP4-MX8`, [Hugging Face](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt) |
| Context | up to 262,144 (longer with YaRN) | up to 1,048,576 | up to 262,144 | up to 1,048,576 | up to 1,048,576 |
| Drafting | MTP, on by default | DSpark, on by default | DFlash 2, on by default | DFly2, on by default | DSpark, on by default |
| Images | yes | yes | yes | yes | yes |
| Documents (PDF, text) | yes | yes | yes | yes | yes |
| Guide | [guide](docs/models/qwen38-flash-next.md) | [guide](docs/models/deepseek41.md) | [guide](docs/models/qwen38-27b.md) | [guide](docs/models/hy4-flash.md) | [guide](docs/models/inkling2.md) |
| Precision | [precision](docs/precision/qwen38-flash-next.md) | [precision](docs/precision/deepseek41.md) | [precision](docs/precision/qwen38-27b.md) | [precision](docs/precision/hy4-flash.md) | [precision](docs/precision/inkling2.md) |

All models compared, with each one's context limit, flags and cards: [models](docs/models/index.md).

## Video

Qwen3.8-Flash-Next and Qwen3.8-27B read video; DeepSeek, Hy4-flash and Inkling 2 Mini do not. A video is named like an image in a chat message, and needs
`--vision` and FFmpeg on the machine. Flash-Next decodes frames on the GPU with NVDEC; the 27B decodes them on the CPU.
How a request names a video, and the limits: [video](docs/running.md#video).

## Documentation

- **Run:** [running](docs/running.md), [configuration](docs/configuration/index.md), [models and the router](docs/configuration/models.md),
  [sampling](docs/configuration/sampling.md), [image sources](docs/configuration/image-sources.md),
  [documents](docs/configuration/documents.md),
  [diagnostics](docs/configuration/diagnostics.md), [basalt-probe](docs/probe.md), [platforms](docs/platforms.md).
- **Models:** [models](docs/models/index.md), [Flash-Next](docs/models/qwen38-flash-next.md), [DeepSeek](docs/models/deepseek41.md),
  [Qwen3.8-27B](docs/models/qwen38-27b.md), [Hy4-flash](docs/models/hy4-flash.md), [Inkling 2 Mini](docs/models/inkling2.md),
  [the `.basalt` file](docs/model-file.md), [file format changes](docs/basalt-file-changelog.md).
- **Quality:** [precision](docs/precision/index.md), [method](docs/precision/quality.md).
- **Build and develop:** [building](docs/building.md), [adding a model](docs/adding-a-model.md), [community runs](docs/community.md),
  [changelog](docs/changelog/index.md), [contributing](CONTRIBUTING.md).

## Credits

- **Code:** [Strata](https://github.com/Niko1221/Strata), the project Basalt forks (MIT); [Infernix](https://github.com/Wallawalla47/Infernix)
  (Ian Ranson and contributors, Apache-2.0), whose kernels are adapted; [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp)
  (MIT); and ideas from [NInfer](https://github.com/Neroued/ninfer). Licences: [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
- **Models:** [Qwen](https://huggingface.co/Qwen), [DeepSeek](https://huggingface.co/deepseek-ai), [Tencent](https://huggingface.co/tencent) (Hy4-flash) and Thinking Machines Lab (Inkling 2 Mini), under the licences on their cards;
  [Unsloth](https://huggingface.co/unsloth)'s GGUFs; [NVIDIA](https://huggingface.co/nvidia)'s NVFP4 checkpoints;
  [QUASAR-QAT](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)'s NVFP4 27B (Apache-2.0); [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)'s DFlash 2 drafter.

Thank you to everyone who has [contributed](https://github.com/jesdga95/basalt/graphs/contributors) or
[sponsored](https://github.com/sponsors/jesdga95) the project.

## License

MIT, see [LICENSE](LICENSE).
