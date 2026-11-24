// tools/tp_crossctx_probe.cu - a standalone verdict on two claims the pair-driver law rests on, and the price
// of the alternative transport:
//   1. "no cross-context events": basalt's pair law (forward.cpp) refuses to choreograph the two cards' streams
//      with events, on the basis that CUDA refuses an event recorded in one card's context from a stream in the
//      other's. capy does it (allreduce.h:403-409). Settle it: record/wait in both cross-context directions and
//      print the driver's actual answer. (basalt's pull lanes never need this - every event there is same-context -
//      but if it works, a push/pipeline redesign gains a scheduling tool.)
//   2. the copy-engine rate of cudaMemcpyPeerAsync in each direction - the push transport (publish = a DMA push
//      into the peer's slot, zero SM time on the wire) prices at this number; the exchange kernel prices at the
//      SM-pull rate. Needs two GPUs; exits 77 without them.
#include <chrono>
#include <cstdio>
#include <functional>
#include <cuda_runtime.h>

namespace {
int failures = 0;
void ck(const char* what, cudaError_t e, bool soft = false) {
    if (e == cudaSuccess) return;
    std::printf("%s %s: %s\n", soft ? "note" : "FAIL", what, cudaGetErrorString(e));
    if (!soft) ++failures;
}
double timed(cudaStream_t s, const std::function<void()>& body) {
    const auto t0 = std::chrono::steady_clock::now();
    body();
    ck("streamSync", cudaStreamSynchronize(s));
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 2) {
        std::printf("SKIP: needs two GPUs\n");
        return 77;
    }
    const size_t bytes = size_t(64) << 20;

    ck("setDevice0", cudaSetDevice(0));
    cudaStream_t s0, s0b;
    ck("streamCreate", cudaStreamCreate(&s0));
    ck("streamCreate", cudaStreamCreate(&s0b));
    void *a0 = nullptr, *peer0 = nullptr;
    ck("malloc", cudaMalloc(&a0, bytes));
    ck("mallocPeerSlot0", cudaMalloc(&peer0, bytes));   // device 0's landing buffer for the peer's pushes
    ck("setDevice1", cudaSetDevice(1));
    cudaStream_t s1;
    ck("streamCreate", cudaStreamCreate(&s1));
    void *a1 = nullptr;
    ck("malloc", cudaMalloc(&a1, bytes));
    ck("memset", cudaMemset(a1, 1, bytes));
    ck("setDevice0", cudaSetDevice(0));
    ck("memset", cudaMemset(a0, 2, bytes));
    ck("setDevice1", cudaSetDevice(1));

    // 1a. record a device-1 context's event, wait it on a device-0 stream.
    ck("setDevice1", cudaSetDevice(1));
    cudaEvent_t ev1;
    ck("eventCreate1", cudaEventCreateWithFlags(&ev1, cudaEventDisableTiming));
    ck("memsetAsync1", cudaMemsetAsync(a1, 3, size_t(1) << 20, s1));
    ck("eventRecord1", cudaEventRecord(ev1, s1));
    ck("setDevice0", cudaSetDevice(0));
    const cudaError_t w1 = cudaStreamWaitEvent(s0, ev1, 0);
    std::printf("note: cross-context wait (event on dev1, stream on dev0): %s\n", cudaGetErrorString(w1));
    if (w1 == cudaSuccess) {
        ck("memsetAsync0", cudaMemsetAsync(a0, 4, size_t(1) << 20, s0));
        ck("sync0", cudaStreamSynchronize(s0));
        std::printf("ok: cross-context wait enqueued and completed\n");
    }

    // 1b. the reverse: record a device-0 event on a device-1 stream.
    cudaEvent_t ev0;
    ck("setDevice0", cudaSetDevice(0));
    ck("eventCreate0", cudaEventCreateWithFlags(&ev0, cudaEventDisableTiming));
    ck("setDevice1", cudaSetDevice(1));
    const cudaError_t r0 = cudaEventRecord(ev0, s1);   // ev0 belongs to dev0's context; s1 to dev1's
    std::printf("note: cross-context record (event on dev0, stream on dev1): %s\n", cudaGetErrorString(r0));
    if (r0 == cudaSuccess) ck("sync1", cudaStreamSynchronize(s1));

    // 2. copy-engine rates. Pull: dev1's stream reads dev0's buffer. Push: dev0's stream writes dev1's buffer.
    for (int dir = 0; dir < 2; ++dir) {
        const double us = dir == 0
            ? (ck("setDevice1", cudaSetDevice(1)),
               timed(s1, [&] { ck("memcpyPeer-pull", cudaMemcpyPeerAsync(a1, 1, a0, 0, bytes, s1)); }))
            : (ck("setDevice0", cudaSetDevice(0)),
               timed(s0b, [&] { ck("memcpyPeer-push", cudaMemcpyPeerAsync(peer0, 0, a1, 1, bytes, s0b)); }));
        std::printf("note: %s of %zu MiB: %.0f us (%.2f GB/s)\n", dir == 0 ? "pull (dev1 reads dev0)" : "push (dev0 writes dev1)",
                    bytes >> 20, us, bytes / us / 1000.0);
    }

    ck("eventDestroy", cudaEventDestroy(ev0));
    ck("eventDestroy", cudaEventDestroy(ev1));
    ck("streamDestroy", cudaStreamDestroy(s0));
    ck("streamDestroy", cudaStreamDestroy(s0b));
    ck("setDevice1", cudaSetDevice(1));
    ck("streamDestroy", cudaStreamDestroy(s1));
    std::printf("%s: %d hard failures (the cross-context answers are notes, not failures)\n",
                failures ? "FAIL" : "ok", failures);
    return failures ? 1 : 0;
}
