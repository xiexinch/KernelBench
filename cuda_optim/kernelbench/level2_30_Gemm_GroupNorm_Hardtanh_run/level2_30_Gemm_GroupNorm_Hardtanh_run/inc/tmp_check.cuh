#include <vector>
__global__ void fused_group_norm_hardtanh_kernel_ori(
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

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    int batch_size = in_batch;
    int num_channels = in_channels;
    int num_groups = 16;
    int channels_per_group = num_channels / num_groups;
    float eps = 1e-5;
    float min_val = -2.0;
    float max_val = 2.0;
    
    // Allocate gamma and beta on device (initialized to 1 and 0 respectively)
    T* gamma;
    T* beta;
    cudaMalloc(&gamma, num_channels * sizeof(T));
    cudaMalloc(&beta, num_channels * sizeof(T));
    
    // Initialize gamma to 1 and beta to 0
    std::vector<T> gamma_host(num_channels, 1.0f);
    std::vector<T> beta_host(num_channels, 0.0f);
    cudaMemcpy(gamma, gamma_host.data(), num_channels * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(beta, beta_host.data(), num_channels * sizeof(T), cudaMemcpyHostToDevice);
    
    int num_blocks = batch_size * num_groups;
    
    fused_group_norm_hardtanh_kernel_ori<<<num_blocks, 1, 0, stream>>>(
        input,
        gamma,
        beta,
        output,
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