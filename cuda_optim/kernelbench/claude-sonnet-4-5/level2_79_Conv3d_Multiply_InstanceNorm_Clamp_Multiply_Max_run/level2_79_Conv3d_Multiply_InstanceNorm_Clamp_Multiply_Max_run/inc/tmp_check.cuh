#include <cuda_runtime.h>
#include <float.h>
#include <type_traits>

__global__ void fused_mult_instnorm_clamp_mult_kernel_ori(
    const float* __restrict__ x,
    const float* __restrict__ multiplier,
    float* __restrict__ out,
    int batch_size,
    int channels,
    int spatial_size,
    float clamp_min,
    float clamp_max
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    
    if (b >= batch_size || c >= channels) return;
    
    int offset = (b * channels + c) * spatial_size;
    
    // First pass: multiply and compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = x[offset + i] * multiplier[c];
        sum += val;
    }
    
    // Reduce sum across threads
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    __syncthreads();
    
    // Second pass: compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = x[offset + i] * multiplier[c];
        float diff = val - mean;
        var_sum += diff * diff;
    }
    
    shared_sum[threadIdx.x] = var_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float variance = shared_sum[0] / spatial_size;
    float std = sqrtf(variance + 1e-5f);
    
    // Third pass: normalize, clamp, multiply
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = x[offset + i] * multiplier[c];
        float normalized = (val - mean) / std;
        float clamped = fminf(fmaxf(normalized, clamp_min), clamp_max);
        out[offset + i] = clamped * multiplier[c];
    }
}

__global__ void max_reduce_channel_kernel_ori(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch_size,
    int channels,
    int spatial_size
) {
    int b = blockIdx.x;
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || idx >= spatial_size) return;
    
    float max_val = -FLT_MAX;
    for (int c = 0; c < channels; c++) {
        float val = x[(b * channels + c) * spatial_size + idx];
        max_val = fmaxf(max_val, val);
    }
    
    out[b * spatial_size + idx] = max_val;
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    static_assert(std::is_same<T, float>::value, "Only float type is supported for this kernel");
    
    float* in_ptr = input;
    float* out_ptr = output;
    
    int batch_size = in_batch;
    int channels = in_channels;
    int spatial_size = in_height * in_width;
    
    // Temporary buffer to hold output of first kernel (same shape as input)
    float* temp_buffer;
    cudaMalloc(&temp_buffer, in_elems * sizeof(float));
    
    // Multiplier array for first kernel (initialized to 1.0f)
    float* multiplier;
    cudaMalloc(&multiplier, channels * sizeof(float));
    float* h_mult = new float[channels];
    for (int i = 0; i < channels; ++i) h_mult[i] = 1.0f;
    cudaMemcpyAsync(multiplier, h_mult, channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    delete[] h_mult;
    
    // First kernel: fused_mult_instnorm_clamp_mult
    // Input: [batch_size, channels, spatial_size]
    // Output: [batch_size, channels, spatial_size]
    dim3 grid1(batch_size, channels);
    int threads1 = 256;
    fused_mult_instnorm_clamp_mult_kernel_ori<<<grid1, threads1, 0, stream>>>(
        in_ptr,
        multiplier,
        temp_buffer,
        batch_size,
        channels,
        spatial_size,
        -1.0f,  // clamp_min
        1.0f    // clamp_max
    );
    
    // Second kernel: max_reduce_channel
    // Input: [batch_size, channels, spatial_size] 
    // Output: [batch_size, spatial_size] (channels reduced)
    int threads2 = 256;
    int blocks_y = (spatial_size + threads2 - 1) / threads2;
    dim3 grid2(batch_size, blocks_y);
    max_reduce_channel_kernel_ori<<<grid2, threads2, 0, stream>>>(
        temp_buffer,
        out_ptr,
        batch_size,
        channels,
        spatial_size
    );
    
    cudaFree(temp_buffer);
    cudaFree(multiplier);
}