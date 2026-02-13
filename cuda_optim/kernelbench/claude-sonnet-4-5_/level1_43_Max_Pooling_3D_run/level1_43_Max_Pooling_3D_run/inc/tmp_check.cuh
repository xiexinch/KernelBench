__global__ void maxpool3d_kernel_ori(
    const float* input,
    float* output,
    int batch_size, int channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size, int stride, int padding, int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_d * output_h * output_w;
    
    if (idx < total_elements) {
        // Calculate indices
        int w_out = idx % output_w;
        int h_out = (idx / output_w) % output_h;
        int d_out = (idx / (output_w * output_h)) % output_d;
        int c = (idx / (output_w * output_h * output_d)) % channels;
        int b = idx / (output_w * output_h * output_d * channels);
        
        float max_val = -INFINITY;
        
        // Iterate over the pooling window
        for (int kd = 0; kd < kernel_size; kd++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int d_in = d_out * stride - padding + kd * dilation;
                    int h_in = h_out * stride - padding + kh * dilation;
                    int w_in = w_out * stride - padding + kw * dilation;
                    
                    if (d_in >= 0 && d_in < input_d &&
                        h_in >= 0 && h_in < input_h &&
                        w_in >= 0 && w_in < input_w) {
                        int input_idx = b * (channels * input_d * input_h * input_w) +
                                       c * (input_d * input_h * input_w) +
                                       d_in * (input_h * input_w) +
                                       h_in * input_w +
                                       w_in;
                        max_val = fmaxf(max_val, input[input_idx]);
                    }
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
    int kernel_size = 3;
    int stride = 2;
    int padding = 1;
    int dilation = 3;
    
    int batch_size = in_batch;
    int channels = in_channels;
    int input_d = in_height;
    int input_h = in_channels;
    int input_w = in_width;
    
    int output_d = (input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_h = (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_w = (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    int total_elements = batch_size * channels * output_d * output_h * output_w;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    maxpool3d_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input,
        output,
        batch_size, channels,
        input