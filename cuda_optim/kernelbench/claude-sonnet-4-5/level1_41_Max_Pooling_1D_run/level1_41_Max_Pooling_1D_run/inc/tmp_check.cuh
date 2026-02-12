__global__ void maxpool1d_kernel_ori(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int features,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * features * output_length;
    
    if (idx < total_elements) {
        int out_pos = idx % output_length;
        int feat = (idx / output_length) % features;
        int batch = idx / (output_length * features);
        
        int input_start = out_pos * stride - padding;
        float max_val = -FLT_MAX;
        
        for (int k = 0; k < kernel_size; k++) {
            int input_pos = input_start + k * dilation;
            if (input_pos >= 0 && input_pos < input_length) {
                int input_idx = batch * features * input_length + feat * input_length + input_pos;
                max_val = fmaxf(max_val, input[input_idx]);
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
    int features = in_channels;
    int input_length = in_width;
    int output_length = out_width;
    
    int kernel_size = 8;
    int stride = 1;
    int padding = 4;
    int dilation = 3;
    
    int total_elements = batch_size * features * output_length;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    maxpool1d_kernel_ori<<<blocks, threads, 0, stream>>>(
        input,
        output,
        batch_size,
        features,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );
}