#include <cuda_runtime.h>

template <typename T>
__global__ void fused_residual_ops_kernel_opt(
    const T* conv_out,
    const T* bias,
    T* output,
    int total_size,
    int channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        T conv_val = conv_out[idx];
        T bias_val = bias[c];
        
        // x = conv_out + bias
        T x = conv_val + bias_val;
        // x = x + conv_out
        x = x + conv_val;
        // x = x * conv_out
        x = x * conv_val;
        // x = x + conv_out
        x = x + conv_val;
        
        output[idx] = x;
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
    // Input buffer layout: [conv_out (size: in_elems), bias (size: in_channels)]
    const T* conv_out = input;
    const T* bias = input + in_elems;
    
    int total_size = in_elems;
    int channels = in_channels;
    int spatial_size = in_height * in_width;
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    fused_residual_ops_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        conv_out,
        bias,
        output,
        total_size,
        channels,
        spatial_size
    );
}