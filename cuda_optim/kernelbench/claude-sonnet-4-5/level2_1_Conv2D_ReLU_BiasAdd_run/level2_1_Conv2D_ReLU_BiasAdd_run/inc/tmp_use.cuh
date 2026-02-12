#include <cuda_runtime.h>

__global__ void fused_relu_bias_kernel_opt(const float* x, const float* bias, float* out, 
                                       int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int c = (idx / (height * width)) % channels;
        float val = x[idx];
        val = val > 0.0f ? val : 0.0f;  // ReLU
        val = val + bias[c];  // Bias addition
        out[idx] = val;
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
    // Assumes T is float for this kernel
    const float* x = reinterpret_cast<const float*>(input);
    // Bias is assumed to be stored immediately after input data in memory
    const float* bias = reinterpret_cast<const float*>(input) + in_elems;
    float* out = reinterpret_cast<float*>(output);
    
    int total_elements = in_batch * in_channels * in_height * in_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_relu_bias_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        x, bias, out,
        in_batch, in_channels, in_height, in_width
    );
}