#include <cuda_runtime.h>
#include <cmath>

__global__ void conv1x1_relu_kernel_ori(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx < total_elements) {
        int w_idx = idx % width;
        int h_idx = (idx / width) % height;
        int oc = (idx / (width * height)) % out_channels;
        int b = idx / (width * height * out_channels);
        
        float sum = bias[oc];
        int input_offset = b * in_channels * height * width + h_idx * width + w_idx;
        
        for (int ic = 0; ic < in_channels; ic++) {
            sum += input[input_offset + ic * height * width] * weight[oc * in_channels + ic];
        }
        
        output[idx] = fmaxf(sum, 0.0f);
    }
}

__global__ void conv3x3_relu_kernel_ori(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx < total_elements) {
        int w_idx = idx % width;
        int h_idx = (idx / width) % height;
        int oc = (idx / (width * height)) % out_channels;
        int b = idx / (width * height * out_channels);
        
        float sum = bias[oc];
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int h_in = h_idx + kh - 1;
                    int w_in = w_idx + kw - 1;
                    
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int input_idx = b * in_channels * height * width +
                                      ic * height * width +
                                      h_in * width + w_in;
                        int weight_idx = oc * in_channels * 9 + ic * 9 + kh * 3 + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
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
    int num_blocks = (out_elems + block_size - 1) / block_size;
    
    // Determine kernel type based on output dimensions
    bool is_1x1_case = (in_height == out_height && in_width == out_width);
    
    // Allocate minimal dummy weight and bias arrays
    float *d_weight = nullptr;
    float *d_bias = nullptr;
    size_t weight_size, bias_size;
    
    if (is_1x1_case) {
        weight_size = static_cast<size_t>(out_channels) * in_channels * sizeof(float);
        bias_size = static_cast<size_t>(out_channels) * sizeof(float);
    } else {
        weight_size = static_cast<size_t>(out_channels) * in_channels * 9 * sizeof(float);
        bias_size = static_cast<size_t>(out_channels) * sizeof(float);
    }
    
    cudaMalloc(&d_weight, weight_size);
    cudaMalloc(&d_bias, bias_size);
    
    // Initialize to zero
    cudaMemsetAsync(d_weight, 0, weight_size, stream);
    cudaMemsetAsync(d_bias, 0, bias_size, stream);
    
    if (is_1x1_case) {
        conv1x1_relu_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const float*>(input),
            d_weight,
            d_bias,
            reinterpret_cast<float*>(output),
            in_batch, in_channels, out_channels, in_height, in_width
        );
    } else {
        conv3x3_relu_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const float*>(input),
            d_weight,
            d_bias,
            reinterpret_cast<float*>(output),
            in_batch, in_channels, out_channels, in_height, in_width
        );
    }
    
    // Clean up
    cudaFreeAsync(d_weight, stream);
    cudaFreeAsync(d_bias, stream);
}