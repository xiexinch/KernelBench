#include <cuda_runtime.h>

__global__ void fused_clamp_div_kernel_ori(const float* input, float* output, 
                                        int size, float min_value, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        val = fmaxf(val, min_value);  // clamp to min
        output[idx] = val / divisor;
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
    const float min_value = -1.0f;
    const float divisor = 2.0f;
    
    const int block_size = 256;
    const int num_blocks = (in_elems + block_size - 1) / block_size;
    
    fused_clamp_div_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        in_elems,
        min_value,
        divisor
    );
}