__global__ void layernorm_forward_kernel_opt(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int batch_size,
    int normalized_size,
    float eps
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* x_batch = x + batch_idx * normalized_size;
    float* out_batch = out + batch_idx * normalized_size;
    
    // Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        sum += x_batch[i];
    }
    
    // Reduce within block
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / normalized_size;
    __syncthreads();
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float diff = x_batch[i] - mean;
        var_sum += diff * diff;
    }
    
    shared_sum[threadIdx.x] = var_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sum[0] / normalized_size;
    float inv_std = rsqrtf(variance + eps);
    __syncthreads();
    
    // Normalize and apply affine transformation
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float normalized = (x_batch[i] - mean) * inv_std;
        out_batch[i] = normalized * gamma[i] + beta[i];
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
    int normalized_size = in_elems / batch_size;
    float eps = 1e-5f;

    // Gamma and beta are not passed as arguments, so we allocate and initialize them on device
    // For benchmarking purposes, we assume they are vectors of size `normalized_size`
    T* d_gamma = nullptr;
    T* d_beta = nullptr;
    cudaMalloc(&d_gamma, normalized_size * sizeof(T));
    cudaMalloc(&d_beta, normalized_size * sizeof(T));

    // Initialize gamma to 1 and beta to 0
    T* h_gamma = new T[normalized_size];
    T* h_beta = new T[normalized_size];
    for (int i = 0; i < normalized_size; ++i) {
        h_gamma[i] = static_cast<T>(1.0);
        h_beta[i] = static_cast<T>(0.0);
    }
    cudaMemcpyAsync(d_gamma, h_gamma, normalized_size * sizeof(T), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_beta, h_beta, normalized_size * sizeof(T), cudaMemcpyHostToDevice, stream);

    const int threads = 256;
    const int blocks = batch_size;

    layernorm_forward_kernel_opt<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(d_gamma),
        reinterpret_cast<const float*>(d_beta),
        reinterpret_cast<float*>(output),
        batch_size,
        normalized_size,
        eps
    );

    // Clean up
    delete[] h_gamma;
    delete[] h_beta;
    cudaFree(d_gamma);
    cudaFree(d_beta);
}