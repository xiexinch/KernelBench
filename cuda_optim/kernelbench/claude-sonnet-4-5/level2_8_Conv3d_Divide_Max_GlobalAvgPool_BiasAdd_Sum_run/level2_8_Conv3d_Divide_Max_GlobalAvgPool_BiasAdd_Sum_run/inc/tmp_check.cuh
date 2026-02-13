__global__ void fused_global_avg_bias_sum_kernel_ori(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int spatial_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        float sum = 0.0f;
        
        // For each channel
        for (int c = 0; c < channels; c++) {
            // Compute global average for this channel
            float channel_sum = 0.0f;
            int base_idx = batch_idx * channels * spatial_size + c * spatial_size;
            
            for (int s = 0; s < spatial_size; s++) {
                channel_sum += input[base_idx + s];
            }
            
            float channel_avg = channel_sum / spatial_size;
            
            // Add bias and accumulate to final sum
            sum += channel_avg + bias[c];
        }
        
        output[batch_idx] = sum;
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
    int spatial_size = in_height * in_width;

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    fused_global_avg_bias_sum_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(output), // Note: bias is passed via output pointer per interface constraint
        reinterpret_cast<float*>(output),
        batch_size,
        channels,
        spatial_size
    );
}