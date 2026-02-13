#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

__global__ void fused_maxpool_logsumexp_relu_kernel_opt(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * out_depth * out_height * out_width;
    
    if (idx < total_out) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int d_out = (idx / (out_width * out_height)) % out_depth;
        int b = idx / (out_width * out_height * out_depth);
        
        // MaxPool3d with kernel_size=2, stride=2
        int d_in_start = d_out * 2;
        int h_in_start = h_out * 2;
        int w_in_start = w_out * 2;
        
        float max_val = -FLT_MAX;
        
        // First find max for numerical stability in logsumexp
        for (int c = 0; c < channels; c++) {
            for (int kd = 0; kd < 2; kd++) {
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int d_in = d_in_start + kd;
                        int h_in = h_in_start + kh;
                        int w_in = w_in_start + kw;
                        
                        if (d_in < in_depth && h_in < in_height && w_in < in_width) {
                            int in_idx = b * (channels * in_depth * in_height * in_width) +
                                       c * (in_depth * in_height * in_width) +
                                       d_in * (in_height * in_width) +
                                       h_in * in_width +
                                       w_in;
                            float val = input[in_idx];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                }
            }
        }
        
        // Compute logsumexp across channels after maxpooling
        float sum_exp = 0.0f;
        
        for (int c = 0; c < channels; c++) {
            float channel_max = -FLT_MAX;
            
            // MaxPool for this channel
            for (int kd = 0; kd < 2; kd++) {
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int d_in = d_in_start + kd;
                        int h_in = h_in_start + kh;
                        int w_in = w_in_start + kw;
                        
                        if (d_in < in_depth && h_in < in_height && w_in < in_width) {
                            int in_idx = b * (channels * in_depth * in_height * in_width) +
                                       c * (in_depth * in_height * in_width) +
                                       d_in * (in_height * in_width) +
                                       h_in * in_width +
                                       w_in;
                            float val = input[in_idx];
                            if (val > channel_max) {
                                channel_max = val;
                            }
                        }
                    }
                }
            }
            
            sum_exp += expf(channel_max - max_val);
        }
        
        float logsumexp_val = max_val + logf(sum_exp);
        
        // Apply ReLU
        output[idx] = logsumexp_val > 0.0f ? logsumexp_val : 0.0f;
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
    int batch_size = in_batch;
    int channels = in_channels;
    int in_depth = in_elems / (in_batch * in_channels * in_height * in_width);
    int out_depth = out_elems / (out_batch * out_height * out_width); // out_channels is 1

    const int block_size = 256;
    int total_out = out_batch * out_depth * out_height * out_width;
    int num_blocks = (total_out + block_size - 1) / block_size;

    fused_maxpool_logsumexp_relu_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        batch_size,
        channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width
    );
}