#include <metal_stdlib>
using namespace metal;

// ─── Forward ─────────────────────────────────────────────────────────────────

kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= M) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++)
        sum += A[gid.y * K + k] * B[k * N + gid.x];
    C[gid.y * N + gid.x] = sum;
}

kernel void matmul_relu(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= M) return;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++)
        sum += A[gid.y * K + k] * B[k * N + gid.x];
    C[gid.y * N + gid.x] = sum > 0.0f ? sum : 0.0f;
}

// Tiled matmul variants — one kernel per tile size so the array dimension is
// a compile-time constant in both the runtime and build-time compilation paths.
// The autotuner benchmarks all three and picks the fastest for each shape.

#define TILED_KERNEL(TILE) \
kernel void matmul_tiled_##TILE( \
    device const float* A [[buffer(0)]], \
    device const float* B [[buffer(1)]], \
    device float*       C [[buffer(2)]], \
    constant uint& M      [[buffer(3)]], \
    constant uint& N      [[buffer(4)]], \
    constant uint& K      [[buffer(5)]], \
    uint2 gid [[thread_position_in_grid]], \
    uint2 lid [[thread_position_in_threadgroup]]) \
{ \
    threadgroup float As[TILE][TILE]; \
    threadgroup float Bs[TILE][TILE]; \
    uint row = gid.y, col = gid.x; \
    float sum = 0.0f; \
    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) { \
        As[lid.y][lid.x] = (row < M && t*TILE+lid.x < K) ? A[row*K + t*TILE+lid.x] : 0.0f; \
        Bs[lid.y][lid.x] = (t*TILE+lid.y < K && col < N) ? B[(t*TILE+lid.y)*N + col] : 0.0f; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint k = 0; k < TILE; k++) sum += As[lid.y][k] * Bs[k][lid.x]; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    if (row < M && col < N) C[row * N + col] = sum; \
}

TILED_KERNEL(8)
TILED_KERNEL(16)
TILED_KERNEL(32)

kernel void relu(
    device const float* inp [[buffer(0)]],
    device float*       out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = inp[gid] > 0.0f ? inp[gid] : 0.0f;
}

// One thread per batch row — numerically stable
kernel void softmax(
    device const float* inp    [[buffer(0)]],
    device float*       out    [[buffer(1)]],
    constant uint& batch       [[buffer(2)]],
    constant uint& classes     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return;
    uint b = gid;
    float maxv = inp[b * classes];
    for (uint c = 1; c < classes; c++)
        maxv = max(maxv, inp[b * classes + c]);
    float sum = 0.0f;
    for (uint c = 0; c < classes; c++)
        sum += exp(inp[b * classes + c] - maxv);
    for (uint c = 0; c < classes; c++)
        out[b * classes + c] = exp(inp[b * classes + c] - maxv) / sum;
}

// Single-thread reduction (small batch/classes in practice)
kernel void cross_entropy(
    device float*       output       [[buffer(0)]],
    device const float* input        [[buffer(1)]],
    device const float* ground_truth [[buffer(2)]],
    constant uint& batch             [[buffer(3)]],
    constant uint& classes           [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid > 0) return;
    float total = 0.0f;
    for (uint b = 0; b < batch; b++) {
        float sample = 0.0f;
        for (uint c = 0; c < classes; c++) {
            float pred = input[b * classes + c] + 1e-8f;
            sample += ground_truth[b * classes + c] * log(pred);
        }
        total -= sample;
    }
    output[0] = total / float(batch);
}

kernel void mse(
    device float*       output       [[buffer(0)]],
    device const float* input        [[buffer(1)]],
    device const float* ground_truth [[buffer(2)]],
    constant uint& size              [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid > 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        float diff = input[i] - ground_truth[i];
        sum += diff * diff;
    }
    output[0] = sum / float(size);
}

// ─── Backward ────────────────────────────────────────────────────────────────

kernel void relu_backward(
    device float*       input_grad  [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device const float* output      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    input_grad[gid] += output_grad[gid] * (output[gid] > 0.0f ? 1.0f : 0.0f);
}

// grad_lhs[i,k] += sum_j  output_grad[i,j] * rhs[k,j]
// dispatch (K, M): gid.x = k, gid.y = i
kernel void matmul_backward_lhs(
    device float*       lhs_grad    [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device const float* rhs         [[buffer(2)]],
    constant uint& M                [[buffer(3)]],
    constant uint& K                [[buffer(4)]],
    constant uint& N                [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= K || gid.y >= M) return;
    float sum = 0.0f;
    for (uint j = 0; j < N; j++)
        sum += output_grad[gid.y * N + j] * rhs[gid.x * N + j];
    lhs_grad[gid.y * K + gid.x] += sum;
}

// grad_rhs[k,j] += sum_i  lhs[i,k] * output_grad[i,j]
// dispatch (N, K): gid.x = j, gid.y = k
kernel void matmul_backward_rhs(
    device float*       rhs_grad    [[buffer(0)]],
    device const float* lhs         [[buffer(1)]],
    device const float* output_grad [[buffer(2)]],
    constant uint& M                [[buffer(3)]],
    constant uint& K                [[buffer(4)]],
    constant uint& N                [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= K) return;
    float sum = 0.0f;
    for (uint i = 0; i < M; i++)
        sum += lhs[i * K + gid.y] * output_grad[i * N + gid.x];
    rhs_grad[gid.y * N + gid.x] += sum;
}

// Same as matmul_backward_lhs but weighted by relu mask from fused output
kernel void matmul_relu_backward_lhs(
    device float*       lhs_grad    [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device const float* rhs         [[buffer(2)]],
    device const float* output      [[buffer(3)]],
    constant uint& M                [[buffer(4)]],
    constant uint& K                [[buffer(5)]],
    constant uint& N                [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= K || gid.y >= M) return;
    float sum = 0.0f;
    for (uint j = 0; j < N; j++) {
        float mask = output[gid.y * N + j] > 0.0f ? 1.0f : 0.0f;
        sum += output_grad[gid.y * N + j] * mask * rhs[gid.x * N + j];
    }
    lhs_grad[gid.y * K + gid.x] += sum;
}

kernel void matmul_relu_backward_rhs(
    device float*       rhs_grad    [[buffer(0)]],
    device const float* lhs         [[buffer(1)]],
    device const float* output_grad [[buffer(2)]],
    device const float* output      [[buffer(3)]],
    constant uint& M                [[buffer(4)]],
    constant uint& K                [[buffer(5)]],
    constant uint& N                [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= K) return;
    float sum = 0.0f;
    for (uint i = 0; i < M; i++) {
        float mask = output[i * N + gid.x] > 0.0f ? 1.0f : 0.0f;
        sum += lhs[i * K + gid.y] * output_grad[i * N + gid.x] * mask;
    }
    rhs_grad[gid.y * N + gid.x] += sum;
}

// One thread per batch row — Jacobian-vector product
kernel void softmax_backward(
    device float*       input_grad  [[buffer(0)]],
    device const float* output_grad [[buffer(1)]],
    device const float* output      [[buffer(2)]],
    constant uint& batch            [[buffer(3)]],
    constant uint& classes          [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return;
    uint b = gid;
    float dot = 0.0f;
    for (uint c = 0; c < classes; c++)
        dot += output_grad[b * classes + c] * output[b * classes + c];
    for (uint c = 0; c < classes; c++) {
        float s = output[b * classes + c];
        float g = output_grad[b * classes + c];
        input_grad[b * classes + c] += s * (g - dot);
    }
}

// One thread per element
kernel void cross_entropy_backward(
    device float*       input_grad   [[buffer(0)]],
    device const float* output_grad  [[buffer(1)]],
    device const float* input        [[buffer(2)]],
    device const float* ground_truth [[buffer(3)]],
    constant uint& batch             [[buffer(4)]],
    constant uint& classes           [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch * classes) return;
    float pred = input[gid] + 1e-8f;
    float grad = (-ground_truth[gid] / pred) * output_grad[0] / float(batch);
    input_grad[gid] += grad;
}

kernel void mse_backward(
    device float*       input_grad   [[buffer(0)]],
    device const float* output_grad  [[buffer(1)]],
    device const float* input        [[buffer(2)]],
    device const float* ground_truth [[buffer(3)]],
    constant uint& size              [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    input_grad[gid] += (2.0f * (input[gid] - ground_truth[gid]) / float(size)) * output_grad[0];
}

// ─── Utility ─────────────────────────────────────────────────────────────────

kernel void sgd_update(
    device float*       storage [[buffer(0)]],
    device const float* grad    [[buffer(1)]],
    constant float& lr          [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    storage[gid] -= lr * grad[gid];
}

kernel void zero_buffer(
    device float* buf [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    buf[gid] = 0.0f;
}
