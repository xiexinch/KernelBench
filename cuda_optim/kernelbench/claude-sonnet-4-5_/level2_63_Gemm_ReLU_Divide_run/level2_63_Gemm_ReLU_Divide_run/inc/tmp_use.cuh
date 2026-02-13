#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void fused_bias_relu_div_kernel_opt(T* data, const T* bias, T divisor, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_features;
    
    if (idx < total_size) {
        int col = idx % out_features;
        T val = data[idx] + bias[col];
        val = val > static_cast<T>(0.0f) ? val : static_cast<T>(0.0f);  // ReLU
        data[idx] = val / divisor;
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
    // Map output dimensions to linear layer dimensions
    // batch_size = out_batch
    // out_features = out_height * out_channels * out_width
    int batch_size = out_batch;
    int out_features = out_height * out_channels * out_width;
    int total_size = out_elems;
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    // input contains bias (size: out_features)
    // output contains data (size: batch_size * out_features)
    T divisor = static_cast<T>(2.0f);
    
    fused_bias_relu_div_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        output,
        input,
        divisor,
        batch_size,
        out_features
    );
}