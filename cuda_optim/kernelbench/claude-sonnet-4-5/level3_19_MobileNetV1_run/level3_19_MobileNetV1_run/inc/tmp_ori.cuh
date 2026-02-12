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
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (W * H)) % C;
        int n = idx / (W * H * C);
        
        float val = input[idx];
        float mean = running_mean[c];
        float var = running_var[c];
        float w_val = weight[c];
        float b_val = bias[c];
        
        // BatchNorm
        val = (val - mean) / sqrtf(var + eps);
        val = val * w_val + b_val;
        
        // ReLU
        val = val > 0.0f ? val : 0.0f;
        
        output[idx] = val;
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream,
    T* weight, T* bias, T* running_mean, T* running_var, float eps)
{
    int N = in_batch;
    int C = in_channels;
    int H = in_height;
    int W = in_width;
    
    int total_size = N * C * H * W;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
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