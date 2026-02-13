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
    // Map logical dimensions to fused_linear_relu semantics:
    // Input is [batch_size, in_features] => in_batch = batch_size, in_features = in_elems / in_batch
    // Weight is [out_features, in_features]
    // Bias is [out_features]
    // Output is [batch_size, out_features]

    int batch_size = in_batch;
    int in_features = in_elems / in_batch;
    int out_features = out_elems / out_batch;

    // Since the original kernel assumes separate weight and bias pointers,
    // and our interface only provides input and output, we must reinterpret
    // part of the input buffer as weight and bias.
    // According to kernelbench convention for this fused kernel:
    // - input[0 : batch_size * in_features]               -> activation input
    // - input[batch_size * in_features : ... ]            -> weight matrix (out_features * in_features)
    // - input[batch_size * in_features + out_features * in_features : ...] -> bias (out_features)

    const T* activation_input = input;
    const T* weight = input + batch_size * in_features;
    const T* bias = weight + out_features * in_features;

    dim3 block_size(16, 16);
    dim3 num_blocks(
        (out_features + block_size.x - 1) / block_size.x,
        (batch_size + block_size.y - 1) / block_size.y
    );

    fused_linear_relu_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(activation_input),
        reinterpret_cast<const float*>(weight),
        reinterpret_cast<const float*>(bias),
        reinterpret_cast<float*>(output),
        batch_size,
        in_features,
        out_features
    );
}