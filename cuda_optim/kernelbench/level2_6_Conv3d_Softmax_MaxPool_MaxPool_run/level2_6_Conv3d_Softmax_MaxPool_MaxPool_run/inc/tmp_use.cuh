__global__ void fused_softmax_double_maxpool3d_kernel_opt(
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
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_el