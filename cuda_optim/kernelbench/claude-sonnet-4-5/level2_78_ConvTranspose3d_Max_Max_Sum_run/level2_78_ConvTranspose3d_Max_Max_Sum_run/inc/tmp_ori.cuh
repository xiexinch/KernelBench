#include <climits>
#include <cfloat>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_maxpool_sum_kernel_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_d, int in_h, int in_w,
    int mid_d, int mid_h, int mid_w,
    int out_d, int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * out_d * out_h * out_w;
    
    if (idx < total_out) {
        int b = idx / (out_d * out_h * out_w);
        int remaining = idx % (out_d * out_h * out_w);
        int od = remaining / (out_h * out_w);
        remaining = remaining % (out_h * out_w);
        int oh = remaining / out_w;
        int ow = remaining % out_w;
        
        float sum_val = 0.0f;
        
        // Iterate over all channels
        for (int c = 0; c < channels; c++) {
            // Apply second max pool (kernel=3)
            float max_val = -FLT_MAX;
            
            for (int kd = 0; kd < 3; kd++) {
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int md = od * 3 + kd;
                        int mh = oh * 3 + kh;
                        int mw = ow * 3 + kw;
                        
                        if (md < mid_d && mh < mid_h && mw < mid_w) {
                            // Apply first max pool (kernel=2)
                            float max_val_first = -FLT_MAX;
                            
                            for (int kd2 = 0; kd2 < 2; kd2++) {
                                for (int kh2 = 0; kh2 < 2; kh2++) {
                                    for (int kw2 = 0; kw2 < 2; kw2++) {
                                        int id = md * 2 + kd2;
                                        int ih = mh * 2 + kh2;
                                        int iw = mw * 2 + kw2;
                                        
                                        if (id < in_d && ih < in_h && iw < in_w) {
                                            int in_idx = b * (channels * in_d * in_h * in_w) +
                                                        c * (in_d * in_h * in_w) +
                                                        id * (in_h * in_w) +
                                                        ih * in_w +
                                                        iw;
                                            max_val_first = fmaxf(max_val_first, input[in_idx]);
                                        }
                                    }
                                }
                            }
                            max_val = fmaxf(max_val, max_val_first);
                        }
                    }
                }
            }
            sum_val += max_val;
        }
        
        output[idx] = sum_val;
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
    int in_d = in_height;
    int in_h = in_width;
    int in_w = 1; // assuming 3D input is passed as 4D with last dim = 1

    // Recompute intermediate and output dimensions based on fixed pooling logic
    int mid_d = in_d / 2;
    int mid_h = in_h / 2;
    int mid_w = in_w / 2;

    int out_d = mid_d / 3;
    int out_h = mid_h / 3;
    int out_w = mid_w / 3;

    int total_out = batch_size * out_d * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total_out + block_size - 1) / block_size;

    fused_maxpool_sum_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        batch_size, channels,
        in_d, in_h, in_w,
        mid_d, mid_h, mid_w,
        out_d, out_h, out_w
    );
}