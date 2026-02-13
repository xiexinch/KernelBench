#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

__global__ void fused_bias_scale_sigmoid_kernel_opt(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int batch_size, int channels, int spatial_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        float val = x[idx] + bias[c];
        val = val * scale[c];
        out[idx] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void group_norm_forward_kernel_opt(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int batch_size, int num_groups, int channels, int spatial_size,
    int channels_per_group, float eps) {
    
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || group_idx >= num_groups) return;
    
    int group_size = channels_per_group * spatial_size;
    int start_c = group_idx * channels_per_group;
    
    // Compute mean
    float sum = 0.0f;
    for (int c = start_c; c < start_c + channels_per_group; c++) {
        for (int s = threadIdx.x; s < spatial_size; s += blockDim.x) {
            int idx = batch_idx * channels * spatial_size + c * spatial_size + s;
            sum += x[idx];
        }
    }
    
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / group_size;
    __syncthreads();
    
    // Compute variance
    float var_sum = 0.0f;
    for (int c = start_c; c < start_c + channels_per_group; c++) {
        for (int s = threadIdx.x; s < spatial_size; s += blockDim.x) {
            int idx = batch_idx * channels * spatial_size + c * spatial_size + s;
            float diff = x[idx] - mean;
            var_sum += diff * diff;
        }
    }
    
    shared_sum[threadIdx.x] = var_sum;
    __syncthreads();
    
    // Reduce variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sum[0] / group_size;
    float std_inv = rsqrtf(variance + eps);
    
    // Normalize
    for (int c = start_c; c < start_c + channels_per_group; c++) {
        for (int s = threadIdx.x; s < spatial_size; s += blockDim.x) {
            int idx = batch_idx * channels * spatial_size + c * spatial_size + s;
            float normalized = (x[idx] - mean) * std_inv;
            out[idx] = normalized * gamma[c] + beta[c];
        }
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    // Original kernels use float, so we cast pointers
    float* d_input = reinterpret_cast<float*>(input);
    float* d_output = reinterpret_cast<float*>(output);
    
    int batch_size = in_batch;
    int channels = in_channels;
    int spatial_size = in_height * in_width;
    int total_size = in_elems;
    
    // Allocate intermediate buffer for chained operations
    float* d_intermediate;
    cudaMalloc(&d_intermediate, total_size * sizeof(float));
    
    // Allocate parameter buffers
    float *d_bias, *d_scale, *d_gamma, *d_beta;
    cudaMalloc(&d_bias, channels * sizeof(float));
    cudaMalloc(&d_scale, channels * sizeof(float));
    cudaMalloc(&d_gamma, channels * sizeof(float));
    cudaMalloc(&d_beta, channels * sizeof(float));
    
    // Initialize parameters on host and copy to device
    float* h_bias = (float*)malloc(channels * sizeof(float));
    float* h_scale = (float*)malloc(channels * sizeof(float));
    float* h_gamma = (float*)malloc(channels * sizeof(float));
    float* h_beta = (float*)malloc(channels * sizeof(float));
    
    for (int i = 0; i < channels; i++) {
        h_bias[i] = 0.0f;
        h_scale[i] = 1.0f;
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }
    
    cudaMemcpyAsync(d_bias, h_bias, channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_scale, h_scale, channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_gamma, h_gamma, channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_beta, h_beta, channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    free(h_bias);
    free(h_scale);
    free(h_gamma);
    free(h_beta);
    
    // Launch fused_bias_scale_sigmoid_kernel_opt
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;
    
    fused_bias_scale_sigmoid_kernel_opt<<<blocks, threads, 0, stream>>>(
        d_input, d_bias, d_scale, d_intermediate,
        batch_size, channels, spatial_size
    );
    
    // Launch group_norm_forward_kernel_opt
    int num_groups = 8;  // As per original model configuration
    int channels_per_group = channels / num_groups;
    float eps = 1e-5f;
    
    dim3 gn_blocks(batch_size, num_groups);
    int gn_threads = 256;
    
    group_norm_forward_kernel_opt<<<gn_blocks, gn_threads, 0, stream>>>(
        d_intermediate, d_gamma, d_beta, d_output,
        batch_size, num_groups, channels, spatial_size,
        channels_per_group, eps
    );
    
    // Cleanup temporary allocations
    cudaFree(d_intermediate);
    cudaFree(d_bias);
    cudaFree(d_scale);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}