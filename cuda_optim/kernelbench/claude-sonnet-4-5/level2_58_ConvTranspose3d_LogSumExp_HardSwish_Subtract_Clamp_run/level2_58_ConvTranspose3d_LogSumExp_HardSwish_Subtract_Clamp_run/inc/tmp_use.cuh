#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_logsumexp_hardswish_clamp_kernel_opt(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width
) {
    int spatial_size = depth * height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * spatial_size) {
        int b = idx / spatial_size;
        int spatial_idx = idx % spatial_size;
        
        // Compute LogSumExp across channels
        float max_val = -FLT_MAX;
        for (int c = 0; c < channels; c++) {
            int input_idx = b * channels * spatial_size + c * spatial_size + spatial_idx;
            max_val = fmaxf(max_val, input[input_idx]);
        }
        
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; c++) {
            int input_idx = b * channels * spatial_size + c * spatial_size + spatial_idx;
            sum_exp += expf(input[input_idx] - max_val);
        }
        
        float logsumexp_val = max_val + logf(sum_exp);
        
        // HardSwish: x * sigmoid(x + 3) / 6
        float x_plus_3 = logsumexp_val + 3.0f;
        float sigmoid_val = 1.0f / (1.0f + expf(-x_plus_3));
        float hardswish_val = logsumexp_val * sigmoid_val / 6.0f;
        
        // Subtract bias
        float result = hardswish_val - bias[0];
        
        // Clamp between -1 and 1
        result = fminf(fmaxf(result, -1.0f), 1.0f);
        
        // Write output
        int output_idx = b * spatial_size + spatial_idx;
        output[output_idx] = result;
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
    // Cast to float pointer (kernel is float32 specific)
    float* input_f = reinterpret_cast<float*>(input);
    float* output_f = reinterpret_cast<float*>(output);
    
    // Map 4D tensor dimensions to kernel's expected 5D parameters
    // Treating spatial dimensions as depth=1, height=in_height, width=in_width
    int batch_size = in_batch;
    int channels = in_channels;
    int depth = 1;
    int height = in_height;
    int width = in_width;
    
    int spatial_size = depth * height * width;
    
    const int threads = 256;
    const int blocks = (batch_size * spatial_size + threads - 1) / threads;
    
    // Allocate device memory for bias scalar (default to 0.0f)
    float h_bias = 0.0f;
    float* d_bias;
    cudaMalloc(&d_bias, sizeof(float));
    cudaMemcpy(d_bias, &h_bias, sizeof(float), cudaMemcpyHostToDevice);
    
    fused_logsumexp_hardswish_clamp_kernel_opt<<<blocks, threads, 0, stream>>>(
        input_f,
        d_bias,
        output_f,
        batch_size,
        channels,
        depth,
        height,
        width
    );
    
    cudaFree(d_bias);
}