#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_softmax_sigmoid_kernel_ori(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = batch_size * spatial_size;
    
    if (idx < total_spatial) {
        int batch_idx = idx / spatial_size;
        int spatial_idx = idx % spatial_size;
        
        // Find max for numerical stability
        float max_val = -FLT_MAX;
        for (int c = 0; c < channels; c++) {
            int input_idx = batch_idx * channels * spatial_size + c * spatial_size + spatial_idx;
            max_val = fmaxf(max_val, input[input_idx]);
        }
        
        // Compute exp sum
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; c++) {
            int input_idx = batch_idx * channels * spatial_size + c * spatial_size + spatial_idx;
            sum_exp += expf(input[input_idx] - max_val);
        }
        
        // Compute softmax and apply sigmoid
        for (int c = 0; c < channels; c++) {
            int input_idx = batch_idx * channels * spatial_size + c * spatial_size + spatial_idx;
            float softmax_val = expf(input[input_idx] - max_val) / sum_exp;
            output[input_idx] = 1.0f / (1.0f + expf(-softmax_val));
        }
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
    // Cast to float pointers (kernel is float-specific)
    float* input_f = reinterpret_cast<float*>(input);
    float* output_f = reinterpret_cast<float*>(output);
    
    // Calculate spatial_size from total elements / (batch * channels)
    // This handles 3D spatial dimensions (D*H*W) flattened
    int spatial_size = in_elems / (in_batch * in_channels);
    
    const int threads = 256;
    const int total_spatial = in_batch * spatial_size;
    const int blocks = (total_spatial + threads - 1) / threads;
    
    fused_softmax_sigmoid_kernel_ori<<<blocks, threads, 0, stream>>>(
        input_f,
        output_f,
        in_batch,
        in_channels,
        spatial_size
    );
}