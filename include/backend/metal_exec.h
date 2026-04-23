#ifndef METAL_EXEC_H
#define METAL_EXEC_H

#include <cstddef>

// Metal backend — mirrors gpu_exec.h but targets the M1 GPU via metal-cpp.
// All pointers are float* (pointing into MTL::Buffer shared memory);
// the implementation keeps an internal map from float* → MTL::Buffer*.

void metalMalloc(float** ptr, size_t bytes);
void metalFree(float* ptr);
void metalCopyToDevice(float* d_dst, const float* h_src, size_t bytes);
void metalCopyToHost(float* h_dst, const float* d_src, size_t bytes);
void metalZeroDevice(float* ptr, int size);

// Forward
void metalMatmulDevice(float* C, float* A, float* B, int M, int N, int K);
void metalReluDevice(float* output, float* input, int size);
void metalMatmulReluDevice(float* C, float* A, float* B, int M, int N, int K);
void metalSoftmaxDevice(float* output, float* input, int batch, int classes);
void metalCrossEntropyDevice(float* output, float* input, float* ground_truth, int batch, int classes);
void metalMseDevice(float* output, float* input, float* ground_truth, int size);

// Backward
void metalReluBackwardDevice(float* input_grad, float* output_grad, float* output, int size);
void metalMatmulBackwardDevice(float* lhs_grad, float* rhs_grad, float* output_grad,
                               float* lhs, float* rhs, int M, int K, int N);
void metalMatmulReluBackwardDevice(float* lhs_grad, float* rhs_grad, float* output_grad,
                                   float* lhs, float* rhs, float* output, int M, int K, int N);
void metalSoftmaxBackwardDevice(float* input_grad, float* output_grad, float* output,
                                int batch, int classes);
void metalCrossEntropyBackwardDevice(float* input_grad, float* output_grad,
                                     float* input, float* ground_truth, int batch, int classes);
void metalMseBackwardDevice(float* input_grad, float* output_grad,
                            float* input, float* ground_truth, int size);

// Training
void metalSgdUpdateDevice(float* storage, float* grad, float lr, int size);

#endif
