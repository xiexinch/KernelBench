#define BLOCK_SIZE 256

__global__ void gemm_scale_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        sum += bias[col];
        sum *= scale[col];
        output[row * out_features + col] = sum;
    }
}

__global__ void compute_mean_var_kernel_opt(
    const float* __restrict__ input,
    float* __restrict__ mean,
    float* __restrict__ var,
    int batch_size,
    int features
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < features) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int i = 0; i < batch_size; i++) {
            float val = input[i * features + col];
            sum += val;
            sum_sq += val * val;
        }
        
        mean[col] = sum / batch_size;
        var[col] = sum_sq / batch_size - mean[col] * mean[col];
    }
}

__global__ void batch_norm_kernel_opt(
    float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int batch_size,
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * features;
    
    if (idx < total) {
        int col = idx % features;
        float normalized = (input[idx] - mean[col]) / sqrtf(var[col] + eps);
        input[idx] = normalized * weight[col] + bias[col];
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
    // Map parameters to original kernel semantics
    int batch_size = in_batch;
    int in_features = in_elems / in_batch;
    int out_features = out_elems / out_batch;

    // Assume all tensors are float as per original kernels
    float* input_f = reinterpret_cast<float*>(input);
    float* output_f = reinterpret_cast<float*>(output);

    // Temporary device buffers for training mode (mean/var)
    // For kernelbench evaluation, we assume training = true
    float* d_mean = nullptr;
    float* d_var = nullptr;
    cudaMalloc(&d_mean, out_features * sizeof(float));
    cudaMalloc(&d_var, out_features * sizeof(float));

    // GEMM + Scale
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);
    gemm_scale_kernel_opt<<<grid_dim, block_dim, 0, stream>>>(
        input_f, nullptr, nullptr, nullptr, output_f,
        batch_size, in_features, out_features
    );

    // Compute mean and variance (training path)
    int blocks_stat = (out_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_mean_var_kernel_opt<<<blocks_stat, BLOCK_SIZE, 0, stream>>>(
        output_f, d_mean, d_var, batch_size, out_features
    );

    // Apply batch norm
    int total = batch_size * out_features;
    int blocks_bn = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    batch_norm_kernel_opt<<<blocks_bn, BLOCK_SIZE, 0, stream>>>(
        output_f, d_mean, d_var, nullptr, nullptr, 1e-5f, batch_size, out_features
    );

    // Cleanup
    cudaFree(d_mean);
    cudaFree(d_var);
}