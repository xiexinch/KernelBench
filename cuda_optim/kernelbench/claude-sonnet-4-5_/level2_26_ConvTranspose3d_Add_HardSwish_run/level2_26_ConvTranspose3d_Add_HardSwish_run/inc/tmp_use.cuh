#include <cuda_runtime.h>

__global__ void fused_add_hardswish_kernel_opt(
    const float* __restrict__ conv_out,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = conv_out[idx] + add_input[idx];
        output[idx] = val * hardswish(val);
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    const float* conv_out = reinterpret_cast<const float*>(input);
    const float* add_input = reinterpret_cast<const float*>(input) + in_elems;
    float* out = reinterpret_cast<float*>(output);
    
    const int block_size = 256;
    int num_blocks = (in_elems + block_size - 1) / block_size;
    
    fused_add_hardswish_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        conv_out, add_input, out, in_elems);
}