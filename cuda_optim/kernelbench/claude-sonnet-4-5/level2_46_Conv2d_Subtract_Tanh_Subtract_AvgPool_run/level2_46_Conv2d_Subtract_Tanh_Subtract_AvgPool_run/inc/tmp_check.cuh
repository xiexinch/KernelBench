__global__ void fused_subtract_tanh_subtract_avgpool_kernel_ori(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float subtract1_value,
    const float subtract2_value,
    const int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * channels * out_height * out_width;
    
    if (idx < total_out) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int b = idx / (out_width * out_height * channels);
        
        float sum = 0.0f;
        int count = 0;
        
        int h_start = h_out * kernel_size;
        int w_start = w_out * kernel_size;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_start + kh;
                int w_in = w_start + kw;
                
                if (h_in < in_height && w_in < in_width) {
                    int in_idx = b * (channels * in_height * in_width) + 
                                c * (in_height * in_width) + 
                                h_in * in_width + w_in;
                    
                    float val = input[in_idx];
                    val = val - subtract1_value;
                    val = tanhf(val);
                    val = val - subtract2_value;
                    sum += val;
                    count++;
                }
            }
        }
        
        output[idx] = sum / count;
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
    const int batch_size = in_batch;
    const int channels = in_channels;
    const int kernel_size = in_width / out_width; // assuming square and uniform pooling

    const int block_size = 256;
    const int num_blocks = (out_elems + block_size - 1) / block_size;

    fused_subtract_tanh_subtract_avgpool_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input,
        output,
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        0.5f, // subtract1_value
        0.2f, // subtract2_value
        kernel_size
    );
}