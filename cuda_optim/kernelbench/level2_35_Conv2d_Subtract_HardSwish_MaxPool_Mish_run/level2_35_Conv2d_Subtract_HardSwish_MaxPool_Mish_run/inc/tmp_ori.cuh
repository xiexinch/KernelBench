__global__ void subtract_hardswish_kernel_opt(const float* input, float* output, 
                                          float subtract_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - subtract_val;
        // hardswish: x * relu6(x + 3) / 6
        // relu6(x) = min(max(0, x), 6)
        float relu6_val = fminf(fmaxf(0.0f, x + 3.0f), 6.0f);
        output[idx] = x * relu6_val / 6.0f;
    }
}

__global__ void maxpool2d_mish_kernel_opt(const float* input, float* output,
                                       int batch, int channels, int height, int width,
                                       int out_height, int out_width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * out_height * out_width;
    
    if (idx < total_elements) {
        int ow = idx % out_width;
        int oh = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int b = idx / (out_width * out_height * channels);
        
        // MaxPool
        float max_val = -INFINITY;
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h = oh * kernel_size + kh;
                int w = ow * kernel_size + kw;
                if (h < height && w < width) {
                    int input_idx = b * (channels * height * width) + 
                                   c * (height * width) + 
                                   h * width + w;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        
        // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
        float softplus = log1pf(expf(max_val));
        float mish_val = max_val * tanhf(softplus);
        
        output[idx] = mish_val;
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
    int size = in_elems;
    float subtract_val = 0.5;
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    T* temp_output = nullptr;
    cudaMalloc(&temp_output, in_elems * sizeof(T));
    
    subtract_hardswish_kernel_opt<<<num_blocks, block_size, 0, stream>>>(input, temp_output, subtract_val, size);
    
    int kernel_size = 2;
    int batch = in_batch;
    int channels = in_channels;
    int height = in_height;
    int width = in_width;
    int out_height_pool = height / kernel_size;
    int out_width_pool = width / kernel_size;
    int total_elements = batch * channels * out_height_pool * out_width_pool;
    int num_blocks_pool = (total_elements + block_size - 1) / block_size;
    
    maxpool2d_mish_kernel_opt<<<num_blocks_pool, block_size, 0, stream>>>(
        temp_output, output,
        batch, channels, height, width