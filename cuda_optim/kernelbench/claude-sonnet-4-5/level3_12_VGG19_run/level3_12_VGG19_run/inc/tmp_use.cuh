#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>

__global__ void conv2d_relu_kernel_opt(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = height + 2 * padding - kernel_size + 1;
    int out_w = width + 2 * padding - kernel_size + 1;
    int total = batch_size * out_channels * out_h * out_w;
    
    if (idx < total) {
        int w_out = idx % out_w;
        int h_out = (idx / out_w) % out_h;
        int c_out = (idx / (out_w * out_h)) % out_channels;
        int b = idx / (out_w * out_h * out_channels);
        
        float sum = bias[c_out];
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int h_in = h_out + kh - padding;
                    int w_in = w_out + kw - padding;
                    
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int input_idx = b * (in_channels * height * width) +
                                       c_in * (height * width) +
                                       h_in * width + w_in;
                        int weight_idx = c_out * (in_channels * kernel_size * kernel_size) +
                                        c_in * (kernel_size * kernel_size) +
                                        kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        output[idx] = fmaxf(sum, 0.0f);
    }
}

__global__ void linear_relu_kernel_opt(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    
    if (idx < total) {
        int b = idx / out_features;
        int o = idx % out_features;
        
        float sum = bias[o];
        for (int i = 0; i < in_features; i++) {
            sum += input[b * in_features + i] * weight[o * in_features + i];
        }
        
        output[idx] = fmaxf(sum, 0.0f);
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
    
    if (in_height == 1 && in_width == 1 && out_height == 1 && out_width == 1) {
        int batch_size = in_batch;
        int in_features = in_channels;
        int out_features = out_channels;
        int total = batch_size * out_features;
        int num_blocks = (total + block_size - 1) / block_size;
        
        const float* input_ptr = reinterpret_cast<const float*>(input);
        const float* weight_ptr = input_ptr + in_elems;
        const float* bias_ptr = weight_ptr + (in_features * out_features);
        float* output_ptr = reinterpret_cast<float*>(output);
        
        linear_relu_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_features, out_features
        );
    } else {
        int batch_size = in_batch;
        int in_ch = in_channels;
        int out_ch = out_channels;
        int height = in_height;
        int width = in_width;
        int kernel_size = 3;
        int padding = 1;
        int total = out_elems;
        int num_blocks = (total + block_size - 1) / block_size;
        
        const float* input_ptr = reinterpret_cast<const float*>(input);
        const float* weight_ptr = input_ptr + in_elems;
        const float* bias_ptr = weight_ptr + (in_ch * out_ch * kernel_size * kernel_size);
        float* output_ptr = reinterpret_cast<float*>(output);
        
        conv2d_relu_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_ch, out_ch, height, width, kernel_size, padding
        );
    }
}