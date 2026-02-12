#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_mean_bias_softmax_tanh_scale_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    float scaling_factor
) {
    // Each thread handles one spatial location (b, h, w)
    int spatial_size = height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * spatial_size) {
        int b = idx / spatial_size;
        int hw = idx % spatial_size;
        int h = hw / width;
        int w = hw % width;
        
        // Step 1: Mean pooling over depth
        float temp[64];  // Assuming max 64 channels
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int d = 0; d < depth; d++) {
                int in_idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
                sum += input[in_idx];
            }
            temp[c] = sum / depth;
        }
        
        // Step 2: Add bias
        for (int c = 0; c < channels; c++) {
            temp[c] += bias[c];
        }
        
        // Step 3: Softmax over channels
        float max_val = temp[0];
        for (int c = 1; c < channels; c++) {
            max_val = fmaxf(max_val, temp[c]);
        }
        
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; c++) {
            temp[c] = expf(temp[c] - max_val);
            sum_exp += temp[c];
        }
        
        for (int c = 0; c < channels; c++) {
            temp[c] /= sum_exp;
        }
        
        // Step 4: Tanh activation and Step 5: Scaling
        for (int c = 0; c < channels; c++) {
            temp[c] = tanhf(temp[c]) * scaling_factor;
        }
        
        // Write output (depth=1 now after mean pooling)
        for (int c = 0; c < channels; c++) {
            int out_idx = ((b * channels + c) * 1 + 0) * height * width + h * width + w;
            output[out_idx] = temp[c];
        }
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
    // Calculate depth from input elements (N*C*D*H*W)
    int depth = in_elems / (in_batch * in_channels * in_height * in_width);
    
    // Cast to float pointers (original kernel is float-specific)
    float* input_f = reinterpret_cast<float*>(input);
    float* output_f = reinterpret_cast<float*>(output);
    
    // Allocate and initialize bias to zero
    float* d_bias = nullptr;
    cudaMalloc(&d_bias, in_channels * sizeof(float));
    cudaMemset(d_bias, 0, in_channels * sizeof(float));
    
    float scaling_factor = 2.0f;
    
    int spatial_size = in_height * in_width;
    int total_threads = in_batch * spatial_size;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;
    
    fused_mean_bias_softmax_tanh_scale_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input_f,
        d_bias,
        output_f,
        in_batch,
        in_channels,
        depth,
        in_height,
        in_width,
        scaling_factor
    );
    
    cudaFree(d_bias);
}