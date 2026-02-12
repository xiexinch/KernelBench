#include <cuda_runtime.h>
#include <type_traits>

__global__ void fused_min_add_mul_kernel_ori(
    const float* input,
    const float* bias,
    float* output,
    const float constant_value,
    const float scaling_factor,
    const int batch_size,
    const int channels,
    const int height,
    const int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * height * width;
    
    if (idx < total_size) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int b = idx / (width * height * channels);
        
        float val = input[idx];
        
        // Apply min operation
        val = fminf(val, constant_value);
        
        // Add bias (broadcast across height and width)
        val = val + bias[c];
        
        // Multiply by scaling factor
        val = val * scaling_factor;
        
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
    static_assert(std::is_same<T, float>::value, "This kernel only supports float type");
    
    // Use hardcoded constants from the original model initialization
    const float constant_value = 0.5f;
    const float scaling_factor = 2.0f;
    
    // Allocate temporary bias buffer (per-channel)
    float* bias = nullptr;
    cudaMalloc(&bias, in_channels * sizeof(float));
    
    // Initialize bias to zero
    cudaMemsetAsync(bias, 0, in_channels * sizeof(float), stream);
    
    const int block_size = 256;
    const int num_blocks = (in_elems + block_size - 1) / block_size;
    
    fused_min_add_mul_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        bias,
        reinterpret_cast<float*>(output),
        constant_value,
        scaling_factor,
        in_batch,
        in_channels,
        in_height,
        in_width
    );
    
    // Cleanup
    cudaFree(bias);
}