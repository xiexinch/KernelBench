#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

__device__ __forceinline__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_hardswish_groupnorm_meanpool_kernel_ori(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size,
    int num_groups,
    float eps
) {
    int batch_idx = blockIdx.x;
    int channel_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || channel_idx >= channels) return;
    
    int channels_per_group = channels / num_groups;
    int group_idx = channel_idx / channels_per_group;
    
    extern __shared__ float shared_mem[];
    float* group_means = shared_mem;
    float* group_vars = &shared_mem[num_groups];
    
    // Initialize shared memory
    if (channel_idx < num_groups) {
        group_means[channel_idx] = 0.0f;
        group_vars[channel_idx] = 0.0f;
    }
    __syncthreads();
    
    // Apply HardSwish and accumulate for mean/variance
    int offset = batch_idx * channels * spatial_size + channel_idx * spatial_size;
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    
    for (int i = 0; i < spatial_size; i++) {
        float val = hardswish(input[offset + i]);
        local_sum += val;
        local_sq_sum += val * val;
    }
    
    // Atomic add to group statistics
    atomicAdd(&group_means[group_idx], local_sum);
    atomicAdd(&group_vars[group_idx], local_sq_sum);
    __syncthreads();
    
    // Compute mean and variance for the group
    float mean = group_means[group_idx] / (channels_per_group * spatial_size);
    float variance = group_vars[group_idx] / (channels_per_group * spatial_size) - mean * mean;
    float inv_std = rsqrtf(variance + eps);
    
    // Apply GroupNorm and compute mean over spatial dimensions
    float channel_sum = 0.0f;
    for (int i = 0; i < spatial_size; i++) {
        float val = hardswish(input[offset + i]);
        float normalized = (val - mean) * inv_std * weight[channel_idx] + bias[channel_idx];
        channel_sum += normalized;
    }
    
    // Write mean-pooled output
    output[batch_idx * channels + channel_idx] = channel_sum / spatial_size;
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    static_assert(std::is_same<T, float>::value, "This kernel only supports float type");
    
    int num_groups = 4;
    float eps = 1e-5f;
    
    int batch_size = in_batch;
    int channels = in_channels;
    int spatial_size = in_elems / (in_batch * in_channels);
    
    // Allocate device memory for weight and bias
    float *d_weight = nullptr;
    float *d_bias = nullptr;
    cudaMalloc(&d_weight, channels * sizeof(float));
    cudaMalloc(&d_bias, channels * sizeof(float));
    
    // Initialize weight to 1.0f and bias to 0.0f
    float* h_weight = new float[channels];
    for (int i = 0; i < channels; i++) {
        h_weight[i] = 1.0f;
    }
    cudaMemcpy(d_weight, h_weight, channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_bias, 0, channels * sizeof(float));
    
    delete[] h_weight;
    
    dim3 grid(batch_size);
    dim3 block(channels);
    int shared_mem_size = 2 * num_groups * sizeof(float);
    
    fused_hardswish_groupnorm_meanpool_kernel_ori<<<grid, block, shared_mem_size, stream>>>(
        (const float*)input,
        d_weight,
        d_bias,
        (float*)output,
        batch_size,
        channels,
        spatial_size,
        num_groups,
        eps
    );
    
    cudaFree(d_weight);
    cudaFree(d_bias);
}