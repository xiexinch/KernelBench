__global__ void fused_ops_kernel_opt(
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
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    // Map inputs to kernel arguments:
    // input layout assumed as:
    //   input[0:in_batch*out_features]               -> gemm_out
    //   input[in_batch*out_features : ...]           -> subtract_param (size = out_features)
    //   input[in_batch*out_features + out_features : ...] -> original_x (size = in_batch * in_features)

    // Based on the original kernel signature, we need:
    // - gemm_out:      [batch_size, out_features]
    // - subtract_param:[out_features]
    // - original_x:    [batch_size, in_features]

    // From the test function signature, infer dimensions:
    int batch_size = in_batch; // or out_batch — both should match
    int out_features = in_channels; // inferred from gemm_out shape: [batch, out_features]
    int in_features = out_channels; // inferred from output shape: [batch, in_features]

    const float* gemm_out = reinterpret_cast<const float*>(input);
    const float* subtract_param = reinterpret_cast<const float*>(input + batch_size * out_features);
    const float* original_x = reinterpret_cast<const float*>(input + batch_size * out_features + out_features);

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    fused_ops_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        gemm_out,
        subtract_param,
        original_x,
        reinterpret_cast<float*>(output),
        batch_size,
        out_features,
        in_features
    );
}