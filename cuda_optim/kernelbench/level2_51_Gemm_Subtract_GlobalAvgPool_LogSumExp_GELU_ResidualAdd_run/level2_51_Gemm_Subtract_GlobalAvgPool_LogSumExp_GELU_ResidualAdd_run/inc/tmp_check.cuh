__global__ void fused_ops_kernel_ori(
    const float* gemm_out,
    const float* subtract_param,
    const float* original_x,
    float* output,
    int batch_size,
    int out_features,
    int in_features
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Step 1: Subtract and compute mean (GlobalAvgPool)
        float sum = 0.0f;
        for (int i = 0; i < out_features; i++) {
            float val = gemm_out[batch_idx * out_features + i] - subtract_param[i];
            sum += val;
        }
        float mean_val = sum / out_features;
        
        // Step 2: LogSumExp on single value (dim=1, but only 1 element)
        // logsumexp of single value is just the value itself
        float lse_val = mean_val;
        
        // Step 3: GELU activation
        // GELU(x) = x * Phi(x), where Phi(x) is CDF of standard normal
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x = lse_val;
        float x_cubed = x * x * x;
        float tanh_arg = 0.7978845608f * (x + 0.044715f * x_cubed);
        float tanh_val = tanhf(tanh_arg);
        float gelu_val = 0.5f * x * (1.0f + tanh_val);
        
        // Step 4: ResidualAdd (broadcast [batch_size, 1] + [batch_size, in_features])
        for (int i = 0; i < in_features; i++) {
            output[batch_idx * in_features + i] = gelu_val + original_x[batch_idx * in_features + i];
        }
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
    // This function is not used for this kernel
    // The actual entry point is through fused_ops_cuda
}