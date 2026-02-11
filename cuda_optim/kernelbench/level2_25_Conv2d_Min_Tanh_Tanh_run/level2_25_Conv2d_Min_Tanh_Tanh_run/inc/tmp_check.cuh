__global__ void fused_min_tanh_kernel_ori(const float* input, float* output, 
                                       int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = height * width;
    int total_elements = batch_size * spatial_size;
    
    if (idx < total_elements) {
        int b = idx / spatial_size;
        int hw = idx % spatial_size;
        
        // Find minimum across channels
        float min_val = FLT_MAX;
        for (int c = 0; c < channels; c++) {
            int input_idx = b * channels * spatial_size + c * spatial_size + hw;
            float val = input[input_idx];
            if (val < min_val) {
                min_val = val;
            }
        }
        
        // Apply tanh twice
        float result = tanhf(min_val);
        result = tanhf(result);
        
        output[idx] = result;
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
    int height = in_height;
    int width = in_width;
    
    int total_elements = batch_size * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_min_tanh_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input, 
        output,
        batch_size, channels, height, width
    );
}