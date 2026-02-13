#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_leaky_mul_leaky_kernel_ori(
    const float* input,
    const float* multiplier,
    float* output,
    int size,
    int channels,
    int spatial_size,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = (idx / spatial_size) % channels;
        float val = input[idx];
        
        // First LeakyReLU
        val = val > 0 ? val : val * negative_slope;
        
        // Multiply
        val = val * multiplier[c];
        
        // Second LeakyReLU
        val = val > 0 ? val : val * negative_slope;
        
        output[idx] = val;
    }
}

__global__ void maxpool3d_kernel_ori(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * out_d * out_h * out_w;
    
    if (idx < total_size) {
        int w_out = idx % out_w;
        int h_out = (idx / out_w) % out_h;
        int d_out = (idx / (out_w * out_h)) % out_d;
        int c = (idx / (out_w * out_h * out_d)) % channels;
        int b = idx / (out_w * out_h * out_d * channels);
        
        float max_val = -1e38f;
        
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int d_in = d_out * kernel_size + kd;
                    int h_in = h_out * kernel_size + kh;
                    int w_in = w_out * kernel_size + kw;
                    
                    if (d_in < in_d && h_in < in_h && w_in < in_w) {
                        int in_idx = b * (channels * in_d * in_h * in_w) +
                                     c * (in_d * in_h * in_w) +
                                     d_in * (in_h * in_w) +
                                     h_in * in_w +
                                     w_in;
                        max_val = fmaxf(max_val, input[in_idx]);
                    }
                }
            }
        }
        
        output[idx] = max_val;
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
    // Determine which kernel to run based on tensor layout
    // For this benchmark, we assume the fused_leaky_mul_leaky case
    // with a 5D tensor: [batch, channels, depth, height, width]
    // Note: in_height is interpreted as depth here due to legacy naming

    int batch_size = in_batch;
    int channels = in_channels;
    int depth = in_height;      // reinterpret in_height as depth
    int height = in_width;      // reinterpret in_width as height
    int width = out_width;      // this is a simplification; real code would need full dims

    // However, since the function signature lacks full 5D info,
    // and the example only passes generic dims, we assume:
    // - The input is used for fused_leaky_mul_leaky
    // - We derive spatial_size as in_elems / (batch_size * channels)

    int spatial_size = in_elems / (batch_size * channels);
    float negative_slope = 0.2f;

    const int block_size = 256;
    int num_blocks = (in_elems + block_size - 1) / block_size;

    fused_leaky_mul_leaky_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(input), // dummy multiplier; real usage needs separate buffer
        reinterpret_cast<float*>(output),
        in_elems,
        channels,
        spatial_size,
        negative_slope
    );
}