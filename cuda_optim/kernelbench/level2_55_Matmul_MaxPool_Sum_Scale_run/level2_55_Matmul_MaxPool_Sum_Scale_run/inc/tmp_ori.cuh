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
    int batch_size = in_batch;
    int in_features = in_height;
    int out_features = in_channels;
    float scale_factor = 0.5;
    
    // Allocate temporary buffer for intermediate result
    T* temp_output;
    cudaMalloc(&temp_output, batch_size * (out_features / 2) * sizeof(T));
    
    // First kernel: fused_linear_maxpool
    const int threads = 256;
    const int blocks_x = (out_features / 2 + threads - 1) / threads;
    dim3 blocks(blocks_x, batch_size);
    
    // Assuming weight and bias are passed through in_width and out_width pointers
    // This is a workaround since the interface doesn't provide weight/bias directly
    T* weight = (T*)((uintptr_t)in_width);
    T* bias = (T*)((uintptr_t)out_width);
    
    fused_linear_maxpool_kernel_opt<<<blocks, threads, 0, stream>>>(
        input,
        weight,
        bias,
        temp_output,
        batch_size,
        in_features,
        out_features
    );
    
    // Second kernel: fused_sum_scale
    int features = out_features / 2;
    const int threads2 = 256;
    const int blocks2 = (batch_size + threads2 - 1) / threads2;
    
    fused_sum_scale_kernel_opt<<<blocks2, threads2, 0, stream