#include <cuda_runtime.h>

__global__ void fused_multiply_leaky_relu_kernel_ori(
    const float* input, 
    float* output, 
    float multiplier, 
    float negative_slope, 
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] * multiplier;
        output[idx] = val > 0.0f ? val : val * negative_slope;
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
    // Parameters from original model initialization
    const float multiplier = 2.0f;
    const float negative_slope = 0.1f;
    
    const int block_size = 256;
    const int num_blocks = (in_elems + block_size - 1) / block_size;
    
    fused_multiply_leaky_relu_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        (const float*)input,
        (float*)output,
        multiplier,
        negative_slope,
        in_elems
    );
}