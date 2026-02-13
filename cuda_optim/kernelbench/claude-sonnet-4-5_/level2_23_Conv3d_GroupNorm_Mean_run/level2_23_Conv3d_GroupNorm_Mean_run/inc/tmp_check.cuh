#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void group_norm_mean_kernel_ori(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int num_groups,
    const int channels,
    const int spatial_size,
    const int channels_per_group,
    const float eps
) {
    const int batch_idx = blockIdx.x;
    const int group_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || group_idx >= num_groups) return;
    
    const int group_start_channel = group_idx * channels_per_group;
    const int group_size = channels_per_group * spatial_size;
    
    // Shared memory for reduction
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_sq_sum[BLOCK_SIZE];
    
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    
    // Compute sum and squared sum for this group
    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int channel_offset = idx / spatial_size;
        int spatial_offset = idx % spatial_size;
        int channel = group_start_channel + channel_offset;
        
        int input_idx = batch_idx * channels * spatial_size + 
                       channel * spatial_size + spatial_offset;
        
        float val = input[input_idx];
        local_sum += val;
        local_sq_sum += val * val;
    }
    
    shared_sum[threadIdx.x] = local_sum;
    shared_sq_sum[threadIdx.x] = local_sq_sum;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
            shared_sq_sum[threadIdx.x] += shared_sq_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    __shared__ float group_mean;
    __shared__ float group_std;
    
    if (threadIdx.x == 0) {
        group_mean = shared_sum[0] / group_size;
        float variance = (shared_sq_sum[0] / group_size) - (group_mean * group_mean);
        group_std = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    // Normalize and accumulate for final mean
    __shared__ float normalized_sum[BLOCK_SIZE];
    float local_normalized_sum = 0.0f;
    
    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int channel_offset = idx / spatial_size;
        int spatial_offset = idx % spatial_size;
        int channel = group_start_channel + channel_offset;
        
        int input_idx = batch_idx * channels * spatial_size + 
                       channel * spatial_size + spatial_offset;
        
        float val = input[input_idx];
        float normalized = (val - group_mean) * group_std;
        normalized = normalized * weight[channel] + bias[channel];
        local_normalized_sum += normalized;
    }
    
    normalized_sum[threadIdx.x] = local_normalized_sum;
    __syncthreads();
    
    // Final reduction for mean
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            normalized_sum[threadIdx.x] += normalized_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(&output[batch_idx], normalized_sum[0]);
    }
}

__global__ void init_ones_kernel_ori(float* ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = 1.0f;
    }
}

__global__ void divide_by_scalar_kernel_ori(float* data, int n, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] /= divisor;
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
    static_assert(sizeof(T) == sizeof(float), "This kernel only supports float (32-bit) type");
    
    float* input_f = reinterpret_cast<float*>(input);
    float* output_f = reinterpret_cast<float*>(output);
    
    const int batch_size = in_batch;
    const int channels = in_channels;
    const int spatial_size = in_height * in_width;
    const int num_groups = channels; // Instance normalization: each channel is a group
    const int channels_per_group = 1;
    const float eps = 1e-5f;
    
    // Initialize output to zero (required for atomicAdd)
    cudaMemsetAsync(output_f, 0, batch_size * sizeof(float), stream);
    
    // Allocate and initialize weight (ones) and bias (zeros)
    float *d_weight, *d_bias;
    cudaMalloc(&d_weight, channels * sizeof(float));
    cudaMalloc(&d_bias, channels * sizeof(float));
    
    // Initialize weight to 1.0
    const int block_size = 256;
    int num_blocks = (channels + block_size - 1) / block_size;
    init_ones_kernel_ori<<<num_blocks, block_size, 0, stream>>>(d_weight, channels);
    
    // Initialize bias to 0.0
    cudaMemsetAsync(d_bias, 0, channels * sizeof(float), stream);
    
    // Launch kernel
    dim3 grid(batch_size, num_groups);
    dim3 block(BLOCK_SIZE);
    
    group_norm_mean_kernel_ori<<<grid, block, 0, stream>>>(
        input_f, d_weight, d_bias, output_f,
        batch_size, num_groups, channels, spatial_size, channels_per_group, eps
    );
    
    // Divide by total_size (channels * spatial_size) to get mean
    const float total_size = static_cast<float>(channels * spatial_size);
    num_blocks = (batch_size + block_size - 1) / block_size;
    divide_by_scalar_kernel_ori<<<num_blocks, block_size, 0, stream>>>(output_f, batch_size, total_size);
    
    cudaFree(d_weight);
    cudaFree(d_bias);
}