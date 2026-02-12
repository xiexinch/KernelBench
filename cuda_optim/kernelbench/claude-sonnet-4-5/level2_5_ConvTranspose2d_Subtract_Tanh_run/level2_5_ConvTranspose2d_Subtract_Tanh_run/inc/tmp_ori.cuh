#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__global__ void fused_bias_tanh_kernel_opt(const T* input, const T* bias, 
                                        T* output, int batch_size, int channels, 
                                        int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * height * width;
    
    if (idx < total_size) {
        int c = (idx / (height * width)) % channels;
        T val = input[idx] - bias[c];
        output[idx] = tanh(val);
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
    const int block_size = 256;
    int num_blocks = (in_elems + block_size - 1) / block_size;
    
    // Bias data is assumed to be stored immediately after input data in memory
    const T* bias = input + in_elems;
    
    fused_bias_tanh_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input,
        bias,
        output,
        in_batch, in_channels, in_height, in_width
    );
}