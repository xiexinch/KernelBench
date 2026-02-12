__global__ void fused_linear_relu_kernel_opt(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        sum += bias[col];
        output[row * out_features + col] = fmaxf(sum, 0.0f);
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
    int in_features = in_height;
    int out_features = out_height;
    
    dim3 block_size(16, 16);
    dim3 num_blocks(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );
    
    fused_linear_relu_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input,
        output,
        output + out_elems,
        output,
        batch_size,
        in_features,
        out_features
    );
}