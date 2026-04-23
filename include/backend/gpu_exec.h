#ifndef EXEC
#define EXEC

#include <cstddef>


void gpuMalloc(float** ptr, size_t size);
void gpuFree(float* ptr);
void gpuCopyToDevice(float* d_dst, const float* h_src, size_t size);
void gpuCopyToHost(float* h_dst, const float* d_src, size_t size);
void zeroDevice(float* d_ptr, int size);

void matmulDevice(float* d_C, float* d_A, float* d_B, int M, int N, int K);
void reluDevice(float* d_output, float* d_input, int size);
void matmulReluDevice(float* d_C, float* d_A, float* d_B, int M, int N, int K);
void softmaxDevice(float* d_output, float* d_input, int batch, int classes);
void crossEntropyDevice(float* d_output, float* d_input, float* d_ground_truth, int batch, int classes);

void mseDevice(float* d_output, float* d_input, float* d_ground_truth, int size);


void reluBackwardDevice(float* d_input_grad, float* d_output_grad, float* d_output, int size);
void matmulBackwardDevice(float* d_lhs_grad, float* d_rhs_grad, float* d_output_grad,
                          float* d_lhs, float* d_rhs, int M, int K, int N);
void matmulReluBackwardDevice(float* d_lhs_grad, float* d_rhs_grad, float* d_output_grad,
                              float* d_lhs, float* d_rhs, float* d_output, int M, int K, int N);
void mseBackwardDevice(float* d_input_grad, float* d_output_grad, float* d_input, float* d_ground_truth, int size);
void softmaxBackwardDevice(float* d_input_grad, float* d_output_grad, float* d_output,
                           int batch, int classes);
void crossEntropyBackwardDevice(float* d_input_grad, float* d_output_grad,
                                float* d_input, float* d_ground_truth, int batch, int classes);

// Training
void sgdUpdateDevice(float* d_storage, float* d_grad, float lr, int size);

#endif
