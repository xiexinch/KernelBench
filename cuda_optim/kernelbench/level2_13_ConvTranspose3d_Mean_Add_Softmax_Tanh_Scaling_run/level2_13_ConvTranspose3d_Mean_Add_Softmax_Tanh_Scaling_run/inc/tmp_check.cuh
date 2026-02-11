__global__ void fused_mean_bias_softmax_tanh_scale_kernel_ori(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    float scaling_factor
) {
    // Each thread handles one spatial location (b, h, w)
    int spatial_size = height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * spatial_size) {
        int b = idx / spatial_size;
        int hw = idx % spatial_size;
        int h = hw / width;
        int w = hw % width;
        
        // Step 1: Mean pooling over depth
        float temp[64];  // Assuming max 64 channels
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int d = 0; d < depth; d++) {
                int in_idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
                sum += input[in_idx];
            }
            temp[c] = sum / depth;
        }
        
        // Step 2: Add bias
        for (int c = 0; c < channels; c++) {
            temp[c] += bias[c];
        }
        
        // Step 3: Softmax over channels
        float max_val = temp[0];
        for (int c = 1; c < channels; c++) {
            max_val = fmaxf(max_val, temp[c]);
        }
        
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; c++) {
            temp[c] = expf(temp[c] - max_val);
            sum_exp += temp[c];
        }
        
        for (int c = 0; c < channels; c++) {
            temp[c] /= sum_exp;
        }
        
        // Step 4: Tanh activation and Step 5: Scaling
        for (int c = 0; c < channels; c++) {
            temp[c] = tanhf(temp[c]) * scaling_factor;
        }
        
        // Write output (depth=1 now after mean pooling)
        for (int c = 0; c < channels; c++) {
            int out_idx = ((b * channels + c) * 1 + 0) * height * width + h * width + w;
            output[out_idx] = temp[c];
        }
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
    int batch_size = in_batch;
    int channels = in_channels;
    int depth = in_height;
    int height = in_width;
    int width = out_width;
    float scaling_factor = 2.0;
    
    T* bias = nullptr;
    cudaMalloc(&bias, channels * sizeof(T));
    
    int spatial_size = height * width;
    int total_threads = batch_size * spatial_size;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;
    
    fused_mean_bias_softmax_tanh_scale_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input,
        bias,
        output,
        batch_size,
        channels,
        depth,
        height,
        width,
        scaling_factor
    );
    
    cudaFree(bias);
}