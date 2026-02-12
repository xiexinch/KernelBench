#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_subtract_mish_kernel_ori(const float* input, float* output, 
                                           float subtract_val_1, float subtract_val_2, 
                                           int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - subtract_val_1 - subtract_val_2;
        // Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        float sp = log1pf(expf(x));  // softplus(x) = ln(1 + e^x)
        output[idx] = x * tanhf(sp);
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
    const float* input_f = reinterpret_cast<const float*>(input);
    float* output_f = reinterpret_cast<float*>(output);
    
    const float subtract_val_1 = 0.5f;
    const float subtract_val_2 = 0.2f;
    
    int size = in_elems;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_subtract_mish_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input_f, 
        output_f, 
        subtract_val_1, 
        subtract_val_2, 
        size
    );
}