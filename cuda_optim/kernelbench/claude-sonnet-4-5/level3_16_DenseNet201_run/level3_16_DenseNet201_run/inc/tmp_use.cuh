__global__ void batchnorm_relu_kernel_opt(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int N, int C, int H, int W,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    
    if (idx < total) {
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
        float normalized = (val - mean) / sqrtf(var + eps);
        float bn_out = normalized * w_val + b_val;
        
        // ReLU
        output[idx] = bn_out > 0 ? bn_out : 0;
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
    // Assume all inputs are float for this kernel
    const float* weight     = reinterpret_cast<const float*>(input + in_elems);
    const float* bias       = reinterpret_cast<const float*>(input + in_elems + in_channels);
    const float* running_mean = reinterpret_cast<const float*>(input + in_elems + 2 * in_channels);
    const float* running_var  = reinterpret_cast<const float*>(input + in_elems + 3 * in_channels);

    float* out_ptr = reinterpret_cast<float*>(output);

    int N = in_batch;
    int C = in_channels;
    int H = in_height;
    int W = in_width;
    float eps = 1e-5f;

    int total = N * C * H * W;
    const int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    batchnorm_relu_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        weight,
        bias,
        running_mean,
        running_var,
        out_ptr,
        N, C, H, W, eps
    );
}