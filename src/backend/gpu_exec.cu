#include "backend/gpu_exec.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>


static cublasHandle_t cublasHandle = nullptr;

static void ensureCublas() {
    if (!cublasHandle) {
        cublasCreate(&cublasHandle);
    }
}

void gpuMalloc(float** ptr, size_t size) {
    if (cudaMalloc(ptr, size) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");
}

void gpuFree(float* ptr) {
    if (ptr) cudaFree(ptr);
}

void gpuCopyToDevice(float* d_dst, const float* h_src, size_t size) {
    if (cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy to device failed");
}

void gpuCopyToHost(float* h_dst, const float* d_src, size_t size) {
    if (cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy to host failed");
}

void zeroDevice(float* d_ptr, int size) {
    cudaMemset(d_ptr, 0, size * sizeof(float));
}

// The kernel (compute function)
__global__ void matmulKernel(float* C, float* A, float* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// The host wrapper function
void matmulDevice(float* d_C, float* d_A, float* d_B, int M, int N, int K) {
    // Configure grid and blocks
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    // Launch kernel
    matmulKernel<<<grid, block>>>(d_C, d_A, d_B, M, N, K);

    // Error checking
    cudaError_t err = cudaGetLastError();
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    // Verify results (print first few elements)
    int peek = (M * N < 8) ? M * N : 8;
    float h_C[8];
}

__global__ void reluKernel(float* out, float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] > 0 ? in[i] : 0;
}

__global__ void reluBackwardKernel(float* in_grad, float* out_grad, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) in_grad[i] += out_grad[i] * (out[i] > 0 ? 1.0f : 0.0f);
}

__global__ void sgdKernel(float* w, float* g, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * g[i];
}

__global__ void softmaxKernel(float* out, float* in, int batch, int classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch) {
        float maxv = -1e38f, sum = 0;
        for (int c = 0; c < classes; c++) maxv = fmaxf(maxv, in[b * classes + c]);
        for (int c = 0; c < classes; c++) sum += expf(in[b * classes + c] - maxv);
        for (int c = 0; c < classes; c++) out[b * classes + c] = expf(in[b * classes + c] - maxv) / sum;
    }
}

__global__ void crossEntropyKernel(float* loss, float* pred, float* target, int batch, int classes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < batch * classes; i++)
            sum -= target[i] * logf(pred[i] + 1e-8f);
        loss[0] = sum / batch;
    }
}

__global__ void crossEntropyBackwardKernel(float* in_grad, float* out_grad, float* pred, float* target, int n, int batch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) in_grad[i] += (-target[i] / (pred[i] + 1e-8f)) * out_grad[0] / batch;
}

__global__ void softmaxBackwardKernel(float* in_grad, float* out_grad, float* out, int batch, int classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch) {
        float dot = 0;
        for (int c = 0; c < classes; c++) dot += out_grad[b * classes + c] * out[b * classes + c];
        for (int c = 0; c < classes; c++) in_grad[b * classes + c] += out[b * classes + c] * (out_grad[b * classes + c] - dot);
    }
}

#define LAUNCH(kernel, n, ...) kernel<<<(n + 255) / 256, 256>>>(__VA_ARGS__)

void reluDevice(float* d_out, float* d_in, int n) { LAUNCH(reluKernel, n, d_out, d_in, n); }
void reluBackwardDevice(float* ig, float* og, float* o, int n) { LAUNCH(reluBackwardKernel, n, ig, og, o, n); }
void sgdUpdateDevice(float* w, float* g, float lr, int n) { LAUNCH(sgdKernel, n, w, g, lr, n); }
void softmaxDevice(float* o, float* i, int b, int c) { LAUNCH(softmaxKernel, b, o, i, b, c); }
void crossEntropyDevice(float* l, float* p, float* t, int b, int c) { crossEntropyKernel<<<1, 1>>>(l, p, t, b, c); }
void crossEntropyBackwardDevice(float* ig, float* og, float* p, float* t, int b, int c) { LAUNCH(crossEntropyBackwardKernel, b*c, ig, og, p, t, b*c, b); }
void softmaxBackwardDevice(float* ig, float* og, float* o, int b, int c) { LAUNCH(softmaxBackwardKernel, b, ig, og, o, b, c); }

void matmulReluDevice(float* d_C, float* d_A, float* d_B, int M, int N, int K) {
    matmulDevice(d_C, d_A, d_B, M, N, K);
    LAUNCH(reluKernel, M * N, d_C, d_C, M * N);
}

__global__ void reluMaskKernel(float* masked_grad, float* grad, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) masked_grad[i] = grad[i] * (output[i] > 0 ? 1.0f : 0.0f);
}

__global__ void mseKernel(float* loss, float* pred, float* target, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            float diff = pred[i] - target[i];
            sum += diff * diff;q
        }
        loss[0] = sum / n;
    }
}

__global__ void mseBackwardKernel(float* in_grad, float* out_grad, float* pred, float* target, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) in_grad[i] += 2.0f * (pred[i] - target[i]) * out_grad[0] / n;
}

void mseDevice(float* l, float* p, float* t, int n) { mseKernel<<<1, 1>>>(l, p, t, n); }
void mseBackwardDevice(float* ig, float* og, float* p, float* t, int n) { LAUNCH(mseBackwardKernel, n, ig, og, p, t, n); }

void matmulBackwardDevice(float* d_lhs_grad, float* d_rhs_grad, float* d_out_grad,
                          float* d_lhs, float* d_rhs, int M, int K, int N) {
    ensureCublas();
    float alpha = 1.0f, beta = 1.0f;
    // grad_lhs = out_grad @ rhs^T
    cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &alpha, d_rhs, N, d_out_grad, N, &beta, d_lhs_grad, K);
    // grad_rhs = lhs^T @ out_grad
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &alpha, d_out_grad, N, d_lhs, K, &beta, d_rhs_grad, N);
}

void matmulReluBackwardDevice(float* d_lhs_grad, float* d_rhs_grad, float* d_out_grad,
                              float* d_lhs, float* d_rhs, float* d_output, int M, int K, int N) {
    int size = M * N;
    float* d_masked_grad;
    cudaMalloc(&d_masked_grad, size * sizeof(float));
    LAUNCH(reluMaskKernel, size, d_masked_grad, d_out_grad, d_output, size);

    ensureCublas();
    float alpha = 1.0f, beta = 1.0f;
    cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &alpha, d_rhs, N, d_masked_grad, N, &beta, d_lhs_grad, K);
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &alpha, d_masked_grad, N, d_lhs, K, &beta, d_rhs_grad, N);

    cudaFree(d_masked_grad);
}