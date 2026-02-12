#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__global__ void fused_min_tanh_kernel_ori(const T* input, T* output, 
                                       int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = height * width;
    int total_elements = batch_size * spatial_size;
    
    if (idx < total_elements) {
        int b = idx / spatial_size;
        int hw = idx % spatial_size;
        
        // Initialize min_val with the first channel to avoid std::numeric_limits in device code
        int base_idx = b * channels * spatial_size + hw;
        T min_val = input[base_idx];
        
        // Find minimum across remaining channels
        for (int c = 1; c < channels; c++) {
            int input_idx = base_idx + c * spatial_size;
            T val = input[input_idx];
            if (val < min_val) {
                min_val = val;
            }
        }
        
        // Apply tanh twice
        T result = tanh(min_val);
        result = tanh(result);
        
        output[idx] = result;
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
    int spatial_size = in_height * in_width;
    int total_elements = in_batch * spatial_size;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_min_tanh_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input, output,
        in_batch, in_channels, in_height, in_width
    );
}