#include <climits>
#include <cfloat>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_min_tanh_kernel_opt(const float* input, float* output, 
                                       int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = height * width;
    int total_elements = batch_size * spatial_size;
    
    if (idx < total_elements) {
        int b = idx / spatial_size;
        int hw = idx % spatial_size;
        
        // Find minimum across channels
        float min_val = FLT_MAX;
        for (int c = 0; c < channels; c++) {
            int input_idx = b * channels * spatial_size + c * spatial_size + hw;
            float val = input[input_idx];
            if (val < min_val) {
                min_val = val;
            }
        }
        
        // Apply tanh twice
        float result = tanhf(min_val);
        result = tanhf(result);
        
        output[idx] = result;
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    const int block_size = 256;
    int total_elements = in_batch * in_height * in_width;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_min_tanh_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        in_batch, in_channels, in_height, in_width
    );
}