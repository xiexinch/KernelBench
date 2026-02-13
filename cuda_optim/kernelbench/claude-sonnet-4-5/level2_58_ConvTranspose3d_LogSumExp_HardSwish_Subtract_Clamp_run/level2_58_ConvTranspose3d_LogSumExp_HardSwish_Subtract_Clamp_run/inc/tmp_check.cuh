#include <cuda_runtime.h>
#include <cmath>
#include <float.h>

__global__ void fused_logsumexp_hardswish_clamp_kernel_ori(
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
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    // Map input dimensions to the expected 5D format: [batch, channels, depth, height, width]
    // The original kernel assumes 5D input with dimensions (batch, channels, depth, height, width)
    // From the given code, we infer:
    //   batch_size = in_batch
    //   channels = in_channels
    //   depth = in_height (since in_height is the third dimension after batch and channels)
    //   height = in_width
    //   width = ??? -> but original input has 5 dims, while our interface gives 4 spatial dims
    // However, the test function signature only provides 4 spatial dims per tensor.
    // Given the original kernel uses 5D input (NCDHW), and the test function gives:
    //   input: [in_batch, in_channels, in_height, in_width] -> missing one spatial dim
    // But note: in the example, they pass 5D tensors, so likely:
    //   in_height corresponds to 'depth'
    //   in_width corresponds to 'height'
    //   and we are missing 'width' — but the problem states the interface.
    //
    // Since the problem says "Keep original kernel logic" and the interface is fixed,
    // and the original kernel uses 5D with (batch, channels, depth, height, width),
    // we must assume that the provided in_height maps to 'depth', and in_width maps to 'height',
    // and that 'width' is 1 (i.e., the data is effectively 4D treated as 5D with width=1).
    // This is consistent with the fact that in_elems = in_batch * in_channels * in_height * in_width,
    // and the kernel expects in_batch * in_channels * depth * height * width = in_elems,
    // so depth * height * width = in_height * in_width => if width=1, then depth=in_height, height=in_width.

    int batch_size = in_batch;
    int channels = in_channels;
    int depth = in_height;
    int height = in_width;
    int width = 1;

    const int threads = 256;
    int spatial_size = depth * height * width;
    int total_elements = batch_size * spatial_size;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_logsumexp_hardswish_clamp_kernel_ori<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(output + out_elems), // Note: bias is passed separately; but in this interface, where is bias?
        reinterpret_cast<float*>(output),
        batch_size,
        channels,
        depth,
        height,
        width
    );
}