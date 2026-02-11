__global__ void batchnorm_relu_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * C * H * W;
    
    if (idx < total_size) {
        int c = (idx / (H * W)) % C;
        
        float mean = running_mean[c];
        float var = running_var[c];
        float w = weight[c];
        float b = bias[c];
        
        float normalized = (input[idx] - mean) / sqrtf(var + eps);
        float scaled = normalized * w + b;
        output[idx] = fmaxf(0.0f, scaled);
    }
}

__global__ void add_relu_kernel_opt(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fmaxf(0.0f, a[idx] + b[idx]);
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
    int N = in_batch;
    int C = in_channels;
    int H = in_height;
    int W = in_width;
    int total_size = N * C * H * W;
    float eps = 1e-5;
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    // Placeholder pointers - in actual usage these would be passed as parameters
    T* weight = nullptr;
    T* bias = nullptr;
    T* running_mean = nullptr;
    T* running_var = nullptr;
    
    batchnorm_relu_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        output,
        N, C, H, W, eps
    );
}