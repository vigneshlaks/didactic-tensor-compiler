#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

#include "../include/metal_exec.h"
#include "../include/autotuner.h"

#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

struct MetalContext {
    MTL::Device*    device = nullptr;
    MTL::CommandQueue* queue = nullptr;
    MTL::Library*   lib    = nullptr;

    MTL::ComputePipelineState* matmulPSO              = nullptr;
    MTL::ComputePipelineState* matmulReluPSO           = nullptr;
    MTL::ComputePipelineState* reluPSO                 = nullptr;
    MTL::ComputePipelineState* softmaxPSO              = nullptr;
    MTL::ComputePipelineState* crossEntropyPSO         = nullptr;
    MTL::ComputePipelineState* msePSO                  = nullptr;
    MTL::ComputePipelineState* reluBackwardPSO         = nullptr;
    MTL::ComputePipelineState* matmulBackwardLhsPSO    = nullptr;
    MTL::ComputePipelineState* matmulBackwardRhsPSO    = nullptr;
    MTL::ComputePipelineState* matmulReluBackwardLhsPSO = nullptr;
    MTL::ComputePipelineState* matmulReluBackwardRhsPSO = nullptr;
    MTL::ComputePipelineState* softmaxBackwardPSO      = nullptr;
    MTL::ComputePipelineState* crossEntropyBackwardPSO = nullptr;
    MTL::ComputePipelineState* mseBackwardPSO          = nullptr;
    MTL::ComputePipelineState* sgdUpdatePSO            = nullptr;
    MTL::ComputePipelineState* zeroBufferPSO           = nullptr;
};

static MetalContext* g_ctx = nullptr;

static std::unordered_map<float*, MTL::Buffer*> g_bufferMap;

static MTL::ComputePipelineState* makePSO(MTL::Device* dev, MTL::Library* lib, const char* name) {
    NS::String* fname = NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function* fn = lib->newFunction(fname);
    if (!fn) {
        throw std::runtime_error(std::string("Metal: function not found: ") + name);
    }
    NS::Error* err = nullptr;
    MTL::ComputePipelineState* pso = dev->newComputePipelineState(fn, &err);
    if (!pso) {
        throw std::runtime_error(std::string("Metal: PSO creation failed for: ") + name);
    }
    fn->release();
    return pso;
}

static MetalContext& getCtx() {
    if (g_ctx) return *g_ctx;

    g_ctx = new MetalContext();
    g_ctx->device = MTLCreateSystemDefaultDevice();
    if (!g_ctx->device) throw std::runtime_error("Metal: no device found");

    std::cout << "[Metal] GPU: " << g_ctx->device->name()->utf8String() << "\n";

    g_ctx->queue = g_ctx->device->newCommandQueue();

    NS::Error* err = nullptr;
#ifdef METALLIB_PATH
    // Xcode available: load the pre-compiled binary (fast, errors caught at build time)
    NS::String* path = NS::String::string(METALLIB_PATH, NS::UTF8StringEncoding);
    NS::URL*    url  = NS::URL::fileURLWithPath(path);
    g_ctx->lib = g_ctx->device->newLibrary(url, &err);
    if (!g_ctx->lib)
        throw std::runtime_error(std::string("Metal: failed to load shaders.metallib: ") +
                                 err->localizedDescription()->utf8String());
#else
    // Command Line Tools only: read shaders.metal and compile at runtime
    std::ifstream file(SHADER_SRC_PATH);
    if (!file.is_open())
        throw std::runtime_error(std::string("Metal: cannot open shader source: ") + SHADER_SRC_PATH);
    std::ostringstream ss;
    ss << file.rdbuf();
    std::string src = ss.str();
    NS::String* srcStr = NS::String::string(src.c_str(), NS::UTF8StringEncoding);
    g_ctx->lib = g_ctx->device->newLibrary(srcStr, nullptr, &err);
    if (!g_ctx->lib)
        throw std::runtime_error(std::string("Metal: shader compile failed: ") +
                                 err->localizedDescription()->utf8String());
#endif

    g_ctx->matmulPSO               = makePSO(g_ctx->device, g_ctx->lib, "matmul");
    g_ctx->matmulReluPSO           = makePSO(g_ctx->device, g_ctx->lib, "matmul_relu");
    g_ctx->reluPSO                 = makePSO(g_ctx->device, g_ctx->lib, "relu");
    g_ctx->softmaxPSO              = makePSO(g_ctx->device, g_ctx->lib, "softmax");
    g_ctx->crossEntropyPSO         = makePSO(g_ctx->device, g_ctx->lib, "cross_entropy");
    g_ctx->msePSO                  = makePSO(g_ctx->device, g_ctx->lib, "mse");
    g_ctx->reluBackwardPSO         = makePSO(g_ctx->device, g_ctx->lib, "relu_backward");
    g_ctx->matmulBackwardLhsPSO    = makePSO(g_ctx->device, g_ctx->lib, "matmul_backward_lhs");
    g_ctx->matmulBackwardRhsPSO    = makePSO(g_ctx->device, g_ctx->lib, "matmul_backward_rhs");
    g_ctx->matmulReluBackwardLhsPSO = makePSO(g_ctx->device, g_ctx->lib, "matmul_relu_backward_lhs");
    g_ctx->matmulReluBackwardRhsPSO = makePSO(g_ctx->device, g_ctx->lib, "matmul_relu_backward_rhs");
    g_ctx->softmaxBackwardPSO      = makePSO(g_ctx->device, g_ctx->lib, "softmax_backward");
    g_ctx->crossEntropyBackwardPSO = makePSO(g_ctx->device, g_ctx->lib, "cross_entropy_backward");
    g_ctx->mseBackwardPSO          = makePSO(g_ctx->device, g_ctx->lib, "mse_backward");
    g_ctx->sgdUpdatePSO            = makePSO(g_ctx->device, g_ctx->lib, "sgd_update");
    g_ctx->zeroBufferPSO           = makePSO(g_ctx->device, g_ctx->lib, "zero_buffer");

    return *g_ctx;
}

static MTL::Buffer* getBuf(float* ptr) {
    auto it = g_bufferMap.find(ptr);
    if (it == g_bufferMap.end())
        throw std::runtime_error("metalExec: unregistered pointer — was metalMalloc called?");
    return it->second;
}

static void dispatch(MTL::ComputePipelineState* pso,
                     std::function<void(MTL::ComputeCommandEncoder*)> setup,
                     MTL::Size threads, MTL::Size tg) {
    auto& ctx = getCtx();
    MTL::CommandBuffer*         cmd = ctx.queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    setup(enc);
    enc->dispatchThreads(threads, tg);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
}

static void dispatch1D(MTL::ComputePipelineState* pso,
                       std::function<void(MTL::ComputeCommandEncoder*)> setup,
                       int n) {
    uint tn = (uint)std::min(n, 256);
    dispatch(pso, setup, MTL::Size(n, 1, 1), MTL::Size(tn, 1, 1));
}

static void dispatch2D(MTL::ComputePipelineState* pso,
                       std::function<void(MTL::ComputeCommandEncoder*)> setup,
                       int w, int h) {
    uint tw = (uint)std::min(w, 16);
    uint th = (uint)std::min(h, 16);
    dispatch(pso, setup, MTL::Size(w, h, 1), MTL::Size(tw, th, 1));
}

void metalMalloc(float** ptr, size_t bytes) {
    auto& ctx = getCtx();
    MTL::Buffer* buf = ctx.device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!buf) throw std::runtime_error("metalMalloc: allocation failed");
    *ptr = static_cast<float*>(buf->contents());
    g_bufferMap[*ptr] = buf;
}

void metalFree(float* ptr) {
    auto it = g_bufferMap.find(ptr);
    if (it != g_bufferMap.end()) {
        it->second->release();
        g_bufferMap.erase(it);
    }
}

// Unified memory — CPU and GPU share the same physical memory on M1
void metalCopyToDevice(float* d_dst, const float* h_src, size_t bytes) {
    std::memcpy(d_dst, h_src, bytes);
}

void metalCopyToHost(float* h_dst, const float* d_src, size_t bytes) {
    std::memcpy(h_dst, d_src, bytes);
}

void metalZeroDevice(float* ptr, int size) {
    auto& ctx = getCtx();
    dispatch1D(ctx.zeroBufferPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(ptr), 0, 0);
    }, size);
}

// Runs matmul_tiled_N for the given tile size and returns wall-clock seconds.
// This is the Metal implementation of BenchmarkFn passed to autotune().
static double metalBenchmarkTile(float* A, float* B, float* C,
                                  int M, int N, int K, int tile) {
    std::string name = "matmul_tiled_" + std::to_string(tile);
    MTL::ComputePipelineState* pso = makePSO(getCtx().device, getCtx().lib, name.c_str());

    uint uM = M, uN = N, uK = K;
    uint gridW = ((uint)N + tile - 1) / tile * tile;
    uint gridH = ((uint)M + tile - 1) / tile * tile;

    auto run = [&]() {
        auto& ctx = getCtx();
        auto t0 = std::chrono::high_resolution_clock::now();
        MTL::CommandBuffer*         cmd = ctx.queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(getBuf(A), 0, 0);
        enc->setBuffer(getBuf(B), 0, 1);
        enc->setBuffer(getBuf(C), 0, 2);
        enc->setBytes(&uM, sizeof(uint), 3);
        enc->setBytes(&uN, sizeof(uint), 4);
        enc->setBytes(&uK, sizeof(uint), 5);
        enc->dispatchThreads(MTL::Size(gridW, gridH, 1), MTL::Size(tile, tile, 1));
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    };

    run();          // warmup — first dispatch includes driver overhead
    double t = run();
    pso->release();
    return t;
}

void metalMatmulDevice(float* C, float* A, float* B, int M, int N, int K) {
    int tile = 0;
    if (!lookupTile(M, N, K, tile))
        tile = autotune(A, B, C, M, N, K, metalBenchmarkTile).bestTile;

    std::string name = "matmul_tiled_" + std::to_string(tile);
    MTL::ComputePipelineState* pso = makePSO(getCtx().device, getCtx().lib, name.c_str());
    uint uM = M, uN = N, uK = K;
    uint gridW = ((uint)N + tile - 1) / tile * tile;
    uint gridH = ((uint)M + tile - 1) / tile * tile;
    dispatch(pso, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(A), 0, 0);
        enc->setBuffer(getBuf(B), 0, 1);
        enc->setBuffer(getBuf(C), 0, 2);
        enc->setBytes(&uM, sizeof(uint), 3);
        enc->setBytes(&uN, sizeof(uint), 4);
        enc->setBytes(&uK, sizeof(uint), 5);
    }, MTL::Size(gridW, gridH, 1), MTL::Size(tile, tile, 1));
    pso->release();
}

void metalReluDevice(float* output, float* input, int size) {
    auto& ctx = getCtx();
    dispatch1D(ctx.reluPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(input),  0, 0);
        enc->setBuffer(getBuf(output), 0, 1);
    }, size);
}

void metalMatmulReluDevice(float* C, float* A, float* B, int M, int N, int K) {
    auto& ctx = getCtx();
    uint uM = M, uN = N, uK = K;
    dispatch2D(ctx.matmulReluPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(A), 0, 0);
        enc->setBuffer(getBuf(B), 0, 1);
        enc->setBuffer(getBuf(C), 0, 2);
        enc->setBytes(&uM, sizeof(uint), 3);
        enc->setBytes(&uN, sizeof(uint), 4);
        enc->setBytes(&uK, sizeof(uint), 5);
    }, N, M);
}

void metalSoftmaxDevice(float* output, float* input, int batch, int classes) {
    auto& ctx = getCtx();
    uint ub = batch, uc = classes;
    // one thread per batch row
    dispatch1D(ctx.softmaxPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(input),  0, 0);
        enc->setBuffer(getBuf(output), 0, 1);
        enc->setBytes(&ub, sizeof(uint), 2);
        enc->setBytes(&uc, sizeof(uint), 3);
    }, batch);
}

void metalCrossEntropyDevice(float* output, float* input, float* ground_truth,
                              int batch, int classes) {
    auto& ctx = getCtx();
    uint ub = batch, uc = classes;
    dispatch1D(ctx.crossEntropyPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(output),       0, 0);
        enc->setBuffer(getBuf(input),        0, 1);
        enc->setBuffer(getBuf(ground_truth), 0, 2);
        enc->setBytes(&ub, sizeof(uint), 3);
        enc->setBytes(&uc, sizeof(uint), 4);
    }, 1);
}

void metalMseDevice(float* output, float* input, float* ground_truth, int size) {
    auto& ctx = getCtx();
    uint us = size;
    dispatch1D(ctx.msePSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(output),       0, 0);
        enc->setBuffer(getBuf(input),        0, 1);
        enc->setBuffer(getBuf(ground_truth), 0, 2);
        enc->setBytes(&us, sizeof(uint), 3);
    }, 1);
}

void metalReluBackwardDevice(float* input_grad, float* output_grad, float* output, int size) {
    auto& ctx = getCtx();
    dispatch1D(ctx.reluBackwardPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(input_grad),  0, 0);
        enc->setBuffer(getBuf(output_grad), 0, 1);
        enc->setBuffer(getBuf(output),      0, 2);
    }, size);
}

void metalMatmulBackwardDevice(float* lhs_grad, float* rhs_grad, float* output_grad,
                               float* lhs, float* rhs, int M, int K, int N) {
    auto& ctx = getCtx();
    uint uM = M, uK = K, uN = N;

    // grad_lhs[M,K]: dispatch (K, M) — gid.x=k, gid.y=i
    dispatch2D(ctx.matmulBackwardLhsPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(lhs_grad),    0, 0);
        enc->setBuffer(getBuf(output_grad), 0, 1);
        enc->setBuffer(getBuf(rhs),         0, 2);
        enc->setBytes(&uM, sizeof(uint), 3);
        enc->setBytes(&uK, sizeof(uint), 4);
        enc->setBytes(&uN, sizeof(uint), 5);
    }, K, M);

    // grad_rhs[K,N]: dispatch (N, K) — gid.x=j, gid.y=k
    dispatch2D(ctx.matmulBackwardRhsPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(rhs_grad),    0, 0);
        enc->setBuffer(getBuf(lhs),         0, 1);
        enc->setBuffer(getBuf(output_grad), 0, 2);
        enc->setBytes(&uM, sizeof(uint), 3);
        enc->setBytes(&uK, sizeof(uint), 4);
        enc->setBytes(&uN, sizeof(uint), 5);
    }, N, K);
}

void metalMatmulReluBackwardDevice(float* lhs_grad, float* rhs_grad, float* output_grad,
                                   float* lhs, float* rhs, float* output, int M, int K, int N) {
    auto& ctx = getCtx();
    uint uM = M, uK = K, uN = N;

    dispatch2D(ctx.matmulReluBackwardLhsPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(lhs_grad),    0, 0);
        enc->setBuffer(getBuf(output_grad), 0, 1);
        enc->setBuffer(getBuf(rhs),         0, 2);
        enc->setBuffer(getBuf(output),      0, 3);
        enc->setBytes(&uM, sizeof(uint), 4);
        enc->setBytes(&uK, sizeof(uint), 5);
        enc->setBytes(&uN, sizeof(uint), 6);
    }, K, M);

    dispatch2D(ctx.matmulReluBackwardRhsPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(rhs_grad),    0, 0);
        enc->setBuffer(getBuf(lhs),         0, 1);
        enc->setBuffer(getBuf(output_grad), 0, 2);
        enc->setBuffer(getBuf(output),      0, 3);
        enc->setBytes(&uM, sizeof(uint), 4);
        enc->setBytes(&uK, sizeof(uint), 5);
        enc->setBytes(&uN, sizeof(uint), 6);
    }, N, K);
}

void metalSoftmaxBackwardDevice(float* input_grad, float* output_grad, float* output,
                                int batch, int classes) {
    auto& ctx = getCtx();
    uint ub = batch, uc = classes;
    dispatch1D(ctx.softmaxBackwardPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(input_grad),  0, 0);
        enc->setBuffer(getBuf(output_grad), 0, 1);
        enc->setBuffer(getBuf(output),      0, 2);
        enc->setBytes(&ub, sizeof(uint), 3);
        enc->setBytes(&uc, sizeof(uint), 4);
    }, batch);
}

void metalCrossEntropyBackwardDevice(float* input_grad, float* output_grad,
                                     float* input, float* ground_truth,
                                     int batch, int classes) {
    auto& ctx = getCtx();
    uint ub = batch, uc = classes;
    dispatch1D(ctx.crossEntropyBackwardPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(input_grad),   0, 0);
        enc->setBuffer(getBuf(output_grad),  0, 1);
        enc->setBuffer(getBuf(input),        0, 2);
        enc->setBuffer(getBuf(ground_truth), 0, 3);
        enc->setBytes(&ub, sizeof(uint), 4);
        enc->setBytes(&uc, sizeof(uint), 5);
    }, batch * classes);
}

void metalMseBackwardDevice(float* input_grad, float* output_grad,
                            float* input, float* ground_truth, int size) {
    auto& ctx = getCtx();
    uint us = size;
    dispatch1D(ctx.mseBackwardPSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(input_grad),   0, 0);
        enc->setBuffer(getBuf(output_grad),  0, 1);
        enc->setBuffer(getBuf(input),        0, 2);
        enc->setBuffer(getBuf(ground_truth), 0, 3);
        enc->setBytes(&us, sizeof(uint), 4);
    }, size);
}

void metalSgdUpdateDevice(float* storage, float* grad, float lr, int size) {
    auto& ctx = getCtx();
    dispatch1D(ctx.sgdUpdatePSO, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(getBuf(storage), 0, 0);
        enc->setBuffer(getBuf(grad),    0, 1);
        enc->setBytes(&lr, sizeof(float), 2);
    }, size);
}
