#include <cuda_runtime.h>

__global__ void double_mish_kernel_ori(const float* __restrict__ input, 
                                   float* __restrict__ output, 
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        val = mish_activation(val);
        val = mish_activation(val);
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
    const int block_size = 256;
    const int num_blocks = (in_elems + block_size - 1) / block_size;
    
    const float* input_f = reinterpret_cast<const float*>(input);
    float* output_f = reinterpret_cast<float*>(output);
    
    double_mish_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input_f, 
        output_f, 
        in_elems
    );
}