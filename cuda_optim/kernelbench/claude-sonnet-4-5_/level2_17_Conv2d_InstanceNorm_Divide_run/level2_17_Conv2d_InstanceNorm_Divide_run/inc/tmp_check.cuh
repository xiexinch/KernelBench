#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

__global__ void fused_instance_norm_div_kernel_ori(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size,
    float divide_by,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels;
    
    if (idx < total) {
        int b = idx / channels;
        int c = idx % channels;
        
        const float* input_ptr = input + (b * channels + c) * spatial_size;
        float* output_ptr = output + (b * channels + c) * spatial_size;
        
        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < spatial_size; i++) {
            sum += input_ptr[i];
        }
        float mean = sum / spatial_size;
        
        // Compute variance
        float var_sum = 0.0f;
        for (int i = 0; i < spatial_size; i++) {
            float diff = input_ptr[i] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / spatial_size;
        float inv_std = rsqrtf(variance + eps);
        
        // Normalize, apply affine transform, and divide
        float g = gamma[c];
        float b_val = beta[c];
        for (int i = 0; i < spatial_size; i++) {
            float normalized = (input_ptr[i] - mean) * inv_std;
            output_ptr[i] = (g * normalized + b_val) / divide_by;
        }
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    // Kernel operates on float data
    int batch_size = in_batch;
    int channels = in_channels;
    int spatial_size = in_height * in_width;
    
    // Allocate device memory for gamma and beta
    float *d_gamma, *d_beta;
    cudaMalloc((void**)&d_gamma, channels * sizeof(float));
    cudaMalloc((void**)&d_beta, channels * sizeof(float));
    
    // Initialize gamma to 1.0f and beta to 0.0f
    float* h_gamma = (float*)malloc(channels * sizeof(float));
    float* h_beta = (float*)malloc(channels * sizeof(float));
    for (int i = 0; i < channels; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }
    cudaMemcpyAsync(d_gamma, h_gamma, channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_beta, h_beta, channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    free(h_gamma);
    free(h_beta);
    
    // Constants for the operation
    float divide_by = 2.0f;
    float eps = 1e-5f;
    
    // Launch kernel
    int total = batch_size * channels;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    
    fused_instance_norm_div_kernel_ori<<<blocks, threads, 0, stream>>>(
        (const float*)input,
        d_gamma,
        d_beta,
        (float*)output,
        batch_size,
        channels,
        spatial_size,
        divide_by,
        eps
    );
    
    // Cleanup temporary allocations
    cudaFree(d_gamma);
    cudaFree(d_beta);
}