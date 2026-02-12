#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

template <typename T>
__global__ void instance_norm_forward_kernel_ori(
    const T* input,
    T* output,
    const int batch_size,
    const int num_features,
    const int height,
    const int width,
    const float eps,
    T* running_mean,
    T* running_var,
    bool track_running_stats,
    bool training) {
    
    const int feature_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int pixel_idx = threadIdx.x + blockIdx.z * blockDim.x;
    
    const int spatial_size = height * width;
    const int total_pixels = spatial_size;
    
    if (feature_idx >= num_features || batch_idx >= batch_size || pixel_idx >= spatial_size) {
        return;
    }
    
    // Shared memory for statistics
    __shared__ T shared_sum[256];
    __shared__ T shared_sqsum[256];
    __shared__ T shared_mean;
    __shared__ T shared_var;
    
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    // Compute mean
    T sum = 0.0f;
    for (int i = tid; i < spatial_size; i += stride) {
        int idx = ((batch_idx * num_features + feature_idx) * spatial_size + i);
        sum += input[idx];
    }
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction for mean
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        shared_mean = shared_sum[0] / spatial_size;
    }
    __syncthreads();
    
    T mean = shared_mean;
    
    // Compute variance
    T sqsum = 0.0f;
    for (int i = tid; i < spatial_size; i += stride) {
        int idx = ((batch_idx * num_features + feature_idx) * spatial_size + i);
        T diff = input[idx] - mean;
        sqsum += diff * diff;
    }
    shared_sqsum[tid] = sqsum;
    __syncthreads();
    
    // Reduction for variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sqsum[tid] += shared_sqsum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        shared_var = shared_sqsum[0] / spatial_size;
    }
    __syncthreads();
    
    T var = shared_var;
    T inv_std = rsqrtf(var + eps);
    
    // Apply normalization
    int output_idx = ((batch_idx * num_features + feature_idx) * spatial_size + pixel_idx);
    if (pixel_idx < spatial_size) {
        output[output_idx] = (input[output_idx] - mean) * inv_std;
    }
    
    // Update running statistics if needed
    if (tid == 0 && track_running_stats && training) {
        int running_idx = feature_idx;
        T momentum = 0.1f;
        
        atomicAdd(&running_mean[running_idx], momentum * (mean - running_mean[running_idx]));
        
        // Use unbiased variance estimate for running variance
        T unbiased_var = (spatial_size < 2) ? var : (var * spatial_size) / (spatial_size - 1);
        atomicAdd(&running_var[running_idx], momentum * (unbiased_var - running_var[running_idx]));
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream) {
    
    const int batch_size = in_batch;
    const int num_features = in_channels;
    const int height = in_height;
    const int width = in_width;
    const int spatial_size = height * width;
    
    // Create dummy running statistics
    T* running_mean;
    T* running_var;
    cudaMalloc(&running_mean, num_features * sizeof(T));
    cudaMalloc(&running_var, num_features * sizeof(T));
    
    // Initialize running statistics
    cudaMemset(running_mean, 0, num_features * sizeof(T));
    cudaMemset(running_var, 0, num_features * sizeof(T));
    
    const float eps = 1e-5f;
    bool track_running_stats = true;
    bool training = true;
    
    // Optimized block and grid configuration
    const int threads_per_block = 256;
    const int blocks_x = num_features;
    const int blocks_y = batch_size;
    const int blocks_z = (spatial_size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_x, blocks_y, blocks_z);
    
    // Launch kernel
    instance_norm_forward_kernel_ori<T><<<grid, threads_per_block, 0, stream>>>(
        input,
        output,
        batch_size,
        num_features,
        height,
        width,
        eps,
        running_mean,
        running_var,
        track_running_stats,
        training);
    
    // Clean up
    cudaFree(running_mean);
    cudaFree(running_var);
}