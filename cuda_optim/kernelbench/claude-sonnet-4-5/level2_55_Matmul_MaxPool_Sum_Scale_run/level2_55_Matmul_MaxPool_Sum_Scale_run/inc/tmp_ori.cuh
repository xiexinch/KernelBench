__global__ void fused_linear_maxpool_kernel_opt(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features / 2) return;
    
    // Compute two consecutive outputs from linear layer
    float val1 = 0.0f;
    float val2 = 0.0f;
    
    int out_idx1 = out_idx * 2;
    int out_idx2 = out_idx * 2 + 1;
    
    // Matrix multiplication
    for (int i = 0; i < in_features; i++) {
        float in_val = input[batch_idx * in_features + i];
        val1 += in_val * weight[out_idx1 * in_features + i];
        val2 += in_val * weight[out_idx2 * in_features + i];
    }
    
    // Add bias
    val1 += bias[out_idx1];
    val2 += bias[out_idx2];
    
    // Max pooling
    float max_val = fmaxf(val1, val2);
    
    output[batch_idx * (out_features / 2) + out_idx] = max_val;
}

__global__ void fused_sum_scale_kernel_opt(
    const float* input,
    float* output,
    int batch_size,
    int features,
    float scale_factor
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    float sum = 0.0f;
    for (int i = 0; i < features; i++) {
        sum += input[batch_idx * features + i];
    }
    
    output[batch_idx] = sum * scale_factor;
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    // Determine which kernel to launch based on tensor shapes
    // Case 1: fused_linear_maxpool (input: [B, in_features], output: [B, out_features/2])
    if (in_height == 1 && in_width == 1 && out_height == 1 && out_width == 1 &&
        in_batch == out_batch && in_elems == in_batch * in_channels &&
        out_elems == out_batch * out_channels) {
        int batch_size = in_batch;
        int in_features = in_channels;
        int out_features = out_channels * 2;

        const float* weight = reinterpret_cast<const float*>(input + in_elems);
        const float* bias = reinterpret_cast<const float*>(input + in_elems + out_features * in_features);

        const int threads = 256;
        const int blocks_x = (out_features / 2 + threads - 1) / threads;
        dim3 blocks(blocks_x, batch_size);

        fused_linear_maxpool_kernel_opt<<<blocks, threads, 0, stream>>>(
            input,
            weight,
            bias,
            output,
            batch_size,
            in_features,
            out_features
        );
    }
    // Case 2: fused_sum_scale (input: [B, features], output: [B])
    else if (in_height == 1 && in_width == 1 && out_height == 1 && out_width == 1 &&
             in_batch == out_batch && out_channels == 1 &&
             in_elems == in_batch * in_channels && out_elems == out_batch) {
        int batch_size = in_batch;
        int features = in_channels;
        float scale_factor = 0.5f; // fixed as per original code

        const int threads = 256;
        const int blocks = (batch_size + threads - 1) / threads;

        fused_sum_scale_kernel_opt<<<blocks, threads, 0, stream>>>(
            input,
            output,
            batch_size,
            features,
            scale_factor
        );
    }
}