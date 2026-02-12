#include <cuda_runtime.h>
#include <float.h>
#include <vector>
#include <type_traits>

__global__ void fused_clamp_softmax_scale_kernel_ori(
    const float* input,
    const float* scale,
    float* output,
    int batch_size,
    int channels,
    int spatial_size,
    float clamp_min,
    float clamp_max
) {
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || channel_idx >= channels) return;
    
    int offset = (batch_idx * channels + channel_idx) * spatial_size;
    const float* input_ptr = input + offset;
    float* output_ptr = output + offset;
    float scale_val = scale[channel_idx];
    
    // Step 1: Clamp and find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = input_ptr[i];
        val = fminf(fmaxf(val, clamp_min), clamp_max);
        max_val = fmaxf(max_val, val);
        output_ptr[i] = val;  // Store clamped value temporarily
    }
    
    // Reduce max across block
    __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, max_val, offset);
        max_val = fmaxf(max_val, other);
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();
    
    // Final reduction
    if (warp_id == 0) {
        max_val = (lane < (blockDim.x + 31) / 32) ? shared_max[lane] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, max_val, offset);
            max_val = fmaxf(max_val, other);
        }
        if (lane == 0) {
            shared_max[0] = max_val;
        }
    }
    __syncthreads();
    max_val = shared_max[0];
    
    // Step 2: Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = expf(output_ptr[i] - max_val);
        output_ptr[i] = val;
        sum += val;
    }
    
    // Reduce sum across block
    __shared__ float shared_sum[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? shared_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            shared_sum[0] = sum;
        }
    }
    __syncthreads();
    sum = shared_sum[0];
    
    // Step 3: Normalize and scale
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        output_ptr[i] = output_ptr[i] * inv_sum * scale_val;
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
    static_assert(std::is_same_v<T, float>, "fused_clamp_softmax_scale_kernel_ori only supports float type");
    
    // Map dimensions: treat spatial dims as flattened (depth * height * width -> height * width)
    int spatial_size = in_height * in_width;
    
    // Default clamp parameters from original implementation
    float clamp_min = 0.0f;
    float clamp_max = 1.0f;
    
    // Allocate temporary scale array initialized to 1.0f (since scale is not provided in signature)
    float* d_scale = nullptr;
    cudaMallocAsync(&d_scale, in_channels * sizeof(float), stream);
    
    std::vector<float> h_scale(in_channels, 1.0f);
    cudaMemcpyAsync(d_scale, h_scale.data(), in_channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // Launch configuration: grid across batch and channels, 256 threads per block
    dim3 grid(in_batch, in_channels);
    int block_size = 256;
    
    fused_clamp_softmax_scale_kernel_ori<<<grid, block_size, 0, stream>>>(
        input,
        d_scale,
        output,
        in_batch,
        in_channels,
        spatial_size,
        clamp_min,
        clamp_max
    );
    
    cudaFreeAsync(d_scale, stream);
}