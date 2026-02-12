__global__ void maxpool2d_kernel_ori(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_height * out_width;
    
    if (idx < total_outputs) {
        int ow = idx % out_width;
        int oh = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int b = idx / (out_width * out_height * channels);
        
        float max_val = -INFINITY;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h = oh * stride - padding + kh * dilation;
                int w = ow * stride - padding + kw * dilation;
                
                if (h >= 0 && h < height && w >= 0 && w < width) {
                    int input_idx = b * (channels * height * width) + 
                                   c * (height * width) + 
                                   h * width + w;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        
        output[idx] = max_val;
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
    int kernel_size = 4;
    int stride = 1;
    int padding = 1;
    int dilation = 1;
    
    int total_outputs = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;
    
    maxpool2d_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input,
        output,
        batch_size,
        channels,
        height,
        width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation
    );
}