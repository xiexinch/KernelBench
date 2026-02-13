#include <cuda_runtime.h>

__global__ void conv1x1_relu_kernel_ori(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx < total_elements) {
        int w_idx = idx % width;
        int h_idx = (idx / width) % height;
        int oc = (idx / (width * height)) % out_channels;
        int b = idx / (width * height * out_channels);
        
        float sum = bias[oc];
        int input_offset = b * in_channels * height * width + h_idx * width + w_idx;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            sum += input[input_offset + ic * height * width] * weight[oc * in_channels + ic];
        }
        
        output[idx] = sum > 0.0f ? sum : 0.0f;  // ReLU
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
    const float* weight = reinterpret_cast<const float*>(input + in_elems);
    const float* bias = reinterpret_cast<const float*>(weight + in_channels * out_channels);
    
    int total_elements = out_batch * out_channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    conv1x1_relu_kernel_ori<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        weight,
        bias,
        reinterpret_cast<float*>(output),
        out_batch, in_channels, out_channels, out_height, out_width
    );
}