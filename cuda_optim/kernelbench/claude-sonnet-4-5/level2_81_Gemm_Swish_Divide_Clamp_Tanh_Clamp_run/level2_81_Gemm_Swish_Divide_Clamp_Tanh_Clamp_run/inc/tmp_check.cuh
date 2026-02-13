#include <cuda_runtime.h>
#include <cmath>

// __device__ float sigmoid(float x) {
//     return 1.0f / (1.0f + expf(-x));
// }

// __device__ float clamp(float x, float min_val, float max_val) {
//     return fmaxf(min_val, fminf(max_val, x));
// }

__global__ void fused_activations_kernel_ori(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        
        // Swish activation: x * sigmoid(x)
        x = x * sigmoid(x);
        
        // Divide by 2.0
        x = x / 2.0f;
        
        // First clamp between -1 and 1
        x = clamp(x, -1.0f, 1.0f);
        
        // Tanh activation
        x = tanhf(x);
        
        // Second clamp between -1 and 1
        x = clamp(x, -1.0f, 1.0f);
        
        output[idx] = x;
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    int size = in_elems;
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    fused_activations_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        size
    );
}