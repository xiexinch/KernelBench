#include <cuda_runtime.h>
#include <type_traits>

__global__ void fused_bias_clamp_scale_kernel_ori(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float scaling_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int c = (idx / (height * width)) % channels;
        int bias_idx = c;
        
        float val = input[idx] + bias[bias_idx];
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val = val * scaling_factor;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val = val / scaling_factor;
        
        output[idx] = val;
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
    static_assert(std::is_same<T, float>::value, "Only float type is supported");
    
    // Static thread-local storage for bias to avoid repeated allocations
    static thread_local float* d_bias = nullptr;
    static thread_local int d_bias_capacity = 0;
    
    // Allocate or reallocate bias buffer if needed
    if (d_bias_capacity < in_channels) {
        if (d_bias != nullptr) {
            cudaFree(d_bias);
        }
        cudaMalloc(&d_bias, in_channels * sizeof(float));
        d_bias_capacity = in_channels;
        // Initialize bias to zero (or any deterministic value)
        cudaMemset(d_bias, 0, in_channels * sizeof(float));
    }
    
    const float scaling_factor = 2.0f;
    const int block_size = 256;
    const int num_blocks = (in_elems + block_size - 1) / block_size;
    
    fused_bias_clamp_scale_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<float*>(input),
        d_bias,
        reinterpret_cast<float*>(output),
        in_batch,
        in_channels,
        in_height,
        in_width,
        scaling_factor
    );
}