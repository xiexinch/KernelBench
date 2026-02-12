__global__ void conv1x1_relu_kernel_ori(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;
    
    if (idx < total_elements) {
        int w_idx = idx % width;
        int h_idx = (idx / width) % height;
        int oc = (idx / (width * height)) % out_channels;
        int b = idx / (width * height * out_channels);
        
        float sum = bias[oc];
        int input_offset = b * in_channels * height * width + h_idx * width + w_idx;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            sum += input[input_offset + ic * height * width] * weight[oc * in_channels + ic];
        }
        
        output[idx] = sum > 0.0f ? sum : 0.0f;  // ReLU
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
    int height = in_height;
    int width = in_width;
    int total_elements = out_elems;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Assuming weight and bias are passed through global memory or constant memory
    // For this entry point, we need to extract them from the input tensor layout
    // Based on the torch code, weight has shape [out_channels, in_channels]
    // and bias has shape [out_channels]
    
    // This assumes input layout: [batch, in_channels, height, width]
    // weight follows after input, bias follows after weight
    const float* weight = input + in_elems;
    const float* bias = weight + (out_channels * in_channels);
    
    conv1x1_relu_kernel_ori<<<blocks, threads, 0, stream>>>(
        input,
        weight,
        bias,
        output,
        batch_size, in_channels, out_channels, height, width
    );
}