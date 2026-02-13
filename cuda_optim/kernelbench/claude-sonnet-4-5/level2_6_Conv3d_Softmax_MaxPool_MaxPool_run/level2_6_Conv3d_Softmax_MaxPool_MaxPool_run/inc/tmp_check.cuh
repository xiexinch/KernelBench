#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

__global__ void fused_softmax_double_maxpool3d_kernel_ori(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    int pool_size,
    int out_depth,
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = depth * height * width;
    int total_elements = batch_size * channels * total_spatial;
    
    if (idx >= batch_size * channels * out_depth * out_height * out_width) return;
    
    int b = idx / (channels * out_depth * out_height * out_width);
    int rem = idx % (channels * out_depth * out_height * out_width);
    int c = rem / (out_depth * out_height * out_width);
    rem = rem % (out_depth * out_height * out_width);
    int od = rem / (out_height * out_width);
    rem = rem % (out_height * out_width);
    int oh = rem / out_width;
    int ow = rem % out_width;
    
    // After two pooling operations with pool_size
    int d_start = od * pool_size * pool_size;
    int h_start = oh * pool_size * pool_size;
    int w_start = ow * pool_size * pool_size;
    
    int pool_size_squared = pool_size * pool_size;
    
    float max_val = -FLT_MAX;
    
    // Apply softmax channel-wise first, then max pool
    for (int pd = 0; pd < pool_size_squared; pd++) {
        for (int ph = 0; ph < pool_size_squared; ph++) {
            for (int pw = 0; pw < pool_size_squared; pw++) {
                int d = d_start + pd;
                int h = h_start + ph;
                int w = w_start + pw;
                
                if (d < depth && h < height && w < width) {
                    // Compute softmax for this spatial location across channels
                    int spatial_idx = b * channels * depth * height * width +
                                     c * depth * height * width +
                                     d * height * width +
                                     h * width +
                                     w;
                    
                    // Get max for numerical stability
                    float max_c = -FLT_MAX;
                    for (int ch = 0; ch < channels; ch++) {
                        int ch_idx = b * channels * depth * height * width +
                                    ch * depth * height * width +
                                    d * height * width +
                                    h * width +
                                    w;
                        max_c = fmaxf(max_c, input[ch_idx]);
                    }
                    
                    // Compute exp sum
                    float sum_exp = 0.0f;
                    for (int ch = 0; ch < channels; ch++) {
                        int ch_idx = b * channels * depth * height * width +
                                    ch * depth * height * width +
                                    d * height * width +
                                    h * width +
                                    w;
                        sum_exp += expf(input[ch_idx] - max_c);
                    }
                    
                    // Compute softmax value for current channel
                    float softmax_val = expf(input[spatial_idx] - max_c) / sum_exp;
                    max_val = fmaxf(max_val, softmax_val);
                }
            }
        }
    }
    
    output[idx] = max_val;
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    // Note: The original kernel assumes 5D input [batch, channels, depth, height, width]
    // The provided signature flattens dimensions, so we must reconstruct depth.
    // From the example usage, in_height is actually 'depth' in the 5D tensor.
    int batch_size = in_batch;
    int channels = in_channels;
    int depth = in_height;     // because in_height corresponds to tensor dimension 2 (depth)
    int height = in_width;     // because in_width corresponds to tensor dimension 3 (height)
    int width = in_elems / (batch_size * channels * depth * height); // infer last dim

    // Recompute output dimensions based on pool_size = 2 (from example)
    int pool_size = 2;
    int pool_size_squared = pool_size * pool_size;
    int computed_out_depth = depth / pool_size_squared;
    int computed_out_height = height / pool_size_squared;
    int computed_out_width = width / pool_size_squared;

    int total_threads = batch_size * channels * computed_out_depth * computed_out_height * computed_out_width;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    fused_softmax_double_maxpool3d_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        batch_size,
        channels,
        depth,
        height,
        width,
        pool_size,
        computed_out_depth,
        computed_out_height,
        computed_out_width
    );
}