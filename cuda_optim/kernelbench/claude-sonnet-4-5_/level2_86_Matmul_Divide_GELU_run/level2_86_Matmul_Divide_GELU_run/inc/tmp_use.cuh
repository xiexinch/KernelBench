#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_div_gelu_kernel_opt(const float* input, float* output, 
                                       int size, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] / divisor;
        // GELU activation: 0.5 * x * (1 + erf(x / sqrt(2)))
        const float inv_sqrt2 = 0.7071067811865475f;
        output[idx] = 0.5f * val * (1.0f + erff(val * inv_sqrt2));
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
    const int block_size = 256;
    const int num_blocks = (in_elems + block_size - 1) / block_size;
    
    float divisor = 1.0f;
    
    fused_div_gelu_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        in_elems,
        divisor
    );
}