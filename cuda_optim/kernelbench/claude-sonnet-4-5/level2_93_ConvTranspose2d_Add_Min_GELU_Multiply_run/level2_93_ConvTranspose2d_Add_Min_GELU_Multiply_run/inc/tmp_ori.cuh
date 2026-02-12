#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_kernel(float x) {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__global__ void fused_add_min_gelu_mul_kernel_opt(
    const float* input,
    float* output,
    const float add_value,
    const float multiply_value,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // Add
        val = val + add_value;
        // Min with 0
        val = fminf(val, 0.0f);
        // GELU
        val = gelu_kernel(val);
        // Multiply
        val = val * multiply_value;
        output[idx] = val;
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
    const float add_value = 0.5f;
    const float multiply_value = 2.0f;
    
    if (in_elems <= 0) return;
    
    const int block_size = 256;
    const int num_blocks = (in_elems + block_size - 1) / block_size;
    
    fused_add_min_gelu_mul_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        add_value,
        multiply_value,
        in_elems
    );
}