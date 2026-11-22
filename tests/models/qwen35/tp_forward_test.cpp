// tests/models/qwen35/tp_forward_test.cpp - the tensor-parallel pair's FIRST forward steps, driven through the
// public session API without the serve wrapper: load the real 27B pack on the pair (--tp 2), prefill a few tokens,
// decode a few windows, and let each call's status plus a cudaGetLastError name the layer whose kernels disagree
// with their card slice.  Prints the greedy picks so the same prompt can be compared against the layer split.
// Needs the pack (argument or BASALT_QWEN35_PACK) and two GPUs; exits 77 without them.
#include "basalt/core/model_file.hpp"
#include "basalt/model/adapter.hpp"
#include "models/qwen35/adapter.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace basalt;

namespace {
int failures = 0;

bool step_ok(const char* what, const model::Result& r, const char* stage) {
    cudaDeviceSynchronize();                    // attribute a device-side fault to the call that queued it
    const cudaError_t e = cudaGetLastError();
    const bool ok = r.ok() && e == cudaSuccess;
    std::printf("%s %s %s: %s%s%s\n", ok ? "ok  " : "FAIL", stage, what, r.message.c_str(),
                e != cudaSuccess ? " | cuda: " : "", e != cudaSuccess ? cudaGetErrorString(e) : "");
    std::fflush(stdout);
    if (!ok) ++failures;
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    const char* env_pack = std::getenv("BASALT_QWEN35_PACK");
    std::string pack = argc > 1 ? argv[1] : (env_pack ? env_pack : "");
    int devs = 0;
    if (pack.empty() || cudaGetDeviceCount(&devs) != cudaSuccess || devs < 2) {
        std::printf("need the 27B pack and two GPUs\n");
        return 77;
    }
    core::ModelFile file;
    std::string err;
    if (!file.open(pack, err) || !file.load_meta(err) || !file.register_gguf_view(err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const std::unique_ptr<model::ModelAdapter> a = models::qwen35::make_adapter();
    model::LoadRequest req;
    req.paths = {pack};
    req.max_context = 8192;
    req.sessions = 1;
    req.tp = 2;
    model::Started started;
    if (const model::Result r = a->start(req, started); !r.ok() || started.sessions.empty()) {
        std::fprintf(stderr, "start: %s\n", r.message.c_str());
        return 1;
    }
    model::ModelSession& sess = *started.sessions.front();
    // a short, ordinary prompt: the chunk path runs, then plain decode windows
    const std::vector<int32_t> prompt{20000, 5000, 1200, 44, 770, 304, 1883, 315, 2570, 66};
    model::PrefillRequest pre;
    pre.tokens = prompt.data();
    pre.n_tokens = (int) prompt.size();
    pre.total_tokens = pre.n_tokens;
    pre.pos = 0;
    if (!step_ok("prefill", sess.prefill_chunk(pre), "chunk")) return failures ? 1 : 0;
    sess.end_prompt();
    int32_t next = prompt.back();
    int64_t pos = (int64_t) prompt.size();
    std::printf("picks:");
    for (int i = 0; i < 8 && !failures; ++i) {
        model::DecodeRequest d;
        d.tokens = &next;
        d.n_tokens = 1;
        d.pos = pos++;
        d.next_token = &next;
        if (!step_ok("decode", sess.decode_window(d), "window")) break;
        std::printf(" %d", next);
        std::fflush(stdout);
    }
    std::printf("\n");
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
