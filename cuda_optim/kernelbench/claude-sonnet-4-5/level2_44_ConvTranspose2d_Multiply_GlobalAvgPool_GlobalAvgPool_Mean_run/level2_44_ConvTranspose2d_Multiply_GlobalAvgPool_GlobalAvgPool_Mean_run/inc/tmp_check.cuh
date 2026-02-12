#include <cuda_runtime.h>

__global__ void fused_mul_gap_kernel_ori(const float* input, float* output, 
                                      float multiplier, int batch_size, 
                                      int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * channels;
    
    if (idx < total_threads) {
        int b = idx / channels;
        int c = idx % channels;
        
        float sum = 0.0f;
        int offset = (b * channels + c) * spatial_size;
        
        for (int i = 0; i < spatial_size; i++) {
            sum += input[offset + i];
        }
        
        output[idx] = (sum / spatial_size) * multiplier;
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
    // Calculate spatial dimensions
    int spatial_size = in_height * in_width;
    int total_threads = in_batch * in_channels;
    
    // Configure kernel launch
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;
    
    // Multiplier value from the original model configuration
    const float multiplier = 0.5f;
    
    // Launch kernel - assumes T is float for this specific kernel
    fused_mul_gap_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        multiplier,
        in_batch,
        in_channels,
        spatial_size
    );
}