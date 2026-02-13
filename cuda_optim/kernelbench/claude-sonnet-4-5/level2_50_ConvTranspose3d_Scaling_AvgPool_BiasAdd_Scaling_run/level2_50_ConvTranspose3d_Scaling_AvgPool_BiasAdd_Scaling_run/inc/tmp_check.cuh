__global__ void fused_scale_avgpool_bias_scale_kernel_ori(
    const float* input,
    float* output,
    const float* bias,
    float scale1,
    float scale2,
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
    int total_elements = batch_size * channels * out_depth * out_height * out_width;
    
    if (idx < total_elements) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int d_out = (idx / (out_width * out_height)) % out_depth;
        int c = (idx / (out_width * out_height * out_depth)) % channels;
        int b = idx / (out_width * out_height * out_depth * channels);
        
        // Average pooling with kernel size 2
        float sum = 0.0f;
        int count = 0;
        
        for (int kd = 0; kd < 2; kd++) {
            for (int kh = 0; kh < 2; kh++) {
                for (int kw = 0; kw < 2; kw++) {
                    int d_in = d_out * 2 + kd;
                    int h_in = h_out * 2 + kh;
                    int w_in = w_out * 2 + kw;
                    
                    if (d_in < in_depth && h_in < in_height && w_in < in_width) {
                        int in_idx = b * (channels * in_depth * in_height * in_width) +
                                     c * (in_depth * in_height * in_width) +
                                     d_in * (in_height * in_width) +
                                     h_in * in_width +
                                     w_in;
                        sum += input[in_idx];
                        count++;
                    }
                }
            }
        }
        
        // Apply operations: scale1, average pool, bias, scale2
        float avg = sum / count;
        float scaled1 = avg * scale1;
        float biased = scaled1 + bias[c];
        float scaled2 = biased * scale2;
        
        output[idx] = scaled2;
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
    // Note: The original kernel assumes 5D input [N, C, D, H, W]
    // The function signature uses generic 4D-like args, but we reinterpret:
    // - in_batch      -> batch_size (N)
    // - in_channels   -> channels (C)
    // - We need to recover in_depth from in_elems and other dims
    // Since in_elems = N * C * D * H * W, we can compute:
    int in_depth = in_elems / (in_batch * in_channels * in_height * in_width);
    int out_depth = out_elems / (out_batch * out_channels * out_height * out_width);

    // Kernel assumes fixed pooling factor of 2, so out_depth = in_depth / 2 etc.
    // But we use the passed dimensions directly.

    const float scale1 = 0.5f;
    const float scale2 = 1.0f;
    const int block_size = 256;
    int total_elements = out_elems;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    // Bias is assumed to be of size [channels], passed via global memory
    // For this benchmark, we assume bias is allocated and passed as part of 'input'
    // However, since the interface doesn't provide bias pointer, we synthesize it:
    // Place bias right after input in memory (as per typical benchmark layout)
    T* bias = input + in_elems;

    fused_scale_avgpool_bias_scale_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        reinterpret_cast<const float*>(bias),
        scale1,
        scale2,
        in_batch,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width
    );
}