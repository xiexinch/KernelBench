#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_group_norm_hardtanh_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int num_channels,
    int num_groups,
    int channels_per_group,
    float eps,
    float min_val,
    float max_val
) {
    int idx = blockIdx.x;
    if (idx >= batch_size * num_groups) return;
    
    int batch_idx = idx / num_groups;
    int group_idx = idx % num_groups;
    
    int start_channel = group_idx * channels_per_group;
    int end_channel = start_channel + channels_per_group;
    
    // Compute mean
    float sum = 0.0f;
    for (int c = start_channel; c < end_channel; c++) {
        sum += input[batch_idx * num_channels + c];
    }
    float mean = sum / channels_per_group;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int c = start_channel; c < end_channel; c++) {
        float diff = input[batch_idx * num_channels + c] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / channels_per_group;
    float inv_std = rsqrtf(variance + eps);
    
    // Normalize, apply affine transformation, and hardtanh
    for (int c = start_channel; c < end_channel; c++) {
        float normalized = (input[batch_idx * num_channels + c] - mean) * inv_std;
        float transformed = normalized * gamma[c] + beta[c];
        // Apply hardtanh
        transformed = fmaxf(min_val, fminf(max_val, transformed));
        output[batch_idx * num_channels + c] = transformed;
    }
}

__global__ void fill_value_kernel_opt(float* data, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    float* input_f = reinterpret_cast<float*>(input);
    float* output_f = reinterpret_cast<float*>(output);
    
    // Configuration from original model
    int batch_size = in_batch;
    int num_channels = in_channels;
    int num_groups = 16;
    float eps = 1e-5f;
    float min_val = -2.0f;
    float max_val = 2.0f;
    
    int channels_per_group = num_channels / num_groups;
    
    // Allocate temporary gamma and beta buffers
    float* gamma = nullptr;
    float* beta = nullptr;
    cudaMalloc(&gamma, num_channels * sizeof(float));
    cudaMalloc(&beta, num_channels * sizeof(float));
    
    // Initialize gamma to 1.0 and beta to 0.0
    const int block_size = 256;
    int init_blocks = (num_channels + block_size - 1) / block_size;
    fill_value_kernel_opt<<<init_blocks, block_size, 0, stream>>>(gamma, num_channels, 1.0f);
    fill_value_kernel_opt<<<init_blocks, block_size, 0, stream>>>(beta, num_channels, 0.0f);
    
    // Launch main kernel with 1 thread per block as in original
    int num_blocks = batch_size * num_groups;
    fused_group_norm_hardtanh_kernel_opt<<<num_blocks, 1, 0, stream>>>(
        input_f,
        gamma,
        beta,
        output_f,
        batch_size,
        num_channels,
        num_groups,
        channels_per_group,
        eps,
        min_val,
        max_val
    );
    
    cudaFree(gamma);
    cudaFree(beta);
}