__global__ void matmul_sigmoid_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int input_size, int hidden_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[row * input_size + i] * weight[col * input_size + i];
        }
        sum += bias[col];
        output[row * hidden_size + col] = 1.0f / (1.0f + expf(-sum));
    }
}

__global__ void matmul_logsumexp_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int hidden_size, int output_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size) {
        float max_val = -INFINITY;
        
        // First pass: find max for numerical stability
        for (int col = 0; col < output_size; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < hidden_size; ++i) {
                sum += input[row * hidden_size + i] * weight[col * hidden_size + i];
            }
            sum += bias[col];
            max_val = fmaxf(max_val, sum);
        }
        
        // Second pass: compute logsumexp
        float exp_sum = 0.0f;
        for (int col = 0; col < output_size; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < hidden_size; ++i) {
                sum += input[row * hidden_size + i] * weight[col * hidden_size + i];
            }
            sum += bias[col];
            exp_sum += expf(sum - max_val);
        }
        
        output[row] = max_val + logf(exp_sum);
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
    int batch_size = in_batch;
    int input_size = in_width;
    int hidden_size = out_width;
    
    dim3 block(16, 16);
    dim3 grid((hidden_size + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
    
    matmul_sigmoid_kernel_opt<<<grid, block, 0, stream>>>(
        input,
        input + in_elems,
        input + in_elems + hidden_size * input_size,
        output,
        batch_size, input_size, hidden_size
    );
}