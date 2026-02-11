__global__ void fused_mul_gap_kernel_opt(const float* input, float* output, 
                                      float multiplier, int batch_size, 
                                      int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * channels;
    
    if (idx < total_threads) {
        int b = idx / channels;
        int c = idx % channels;
        
        float sum = 0.0f;
        int offset = (b * channels + c) * spatial_size;
        
        for (int i = 0; i < spatial_size; i++) {
            sum += input[offset + i];
        }
        
        output[idx] = (sum / spatial_size) * multiplier;
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
    int spatial_size = in_height * in_width;
    float multiplier = 0.5;
    
    const int block_size = 256;
    const int num_blocks = (batch_size * channels + block_size - 1) / block_size;
    
    fused_mul_gap_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input, 
        output,
        multiplier,
        batch_size, 
        channels, 
        spatial_size
    );
}