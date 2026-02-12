#include <vector>
__global__ void layernorm_forward_kernel_ori(
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
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    int batch_size = in_batch;
    int normalized_size = in_elems / batch_size;
    float eps = 1e-5;
    
    // Allocate gamma and beta on device (initialized to 1 and 0 respectively)
    T* gamma;
    T* beta;
    cudaMalloc(&gamma, normalized_size * sizeof(T));
    cudaMalloc(&beta, normalized_size * sizeof(T));
    
    // Initialize gamma to 1 and beta to 0
    std::vector<T> gamma_host(normalized_size, 1.0f);
    std::vector<T> beta_host(normalized_size, 0.0f);
    cudaMemcpy(gamma, gamma_host.data(), normalized_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(beta, beta_host.data(), normalized_size * sizeof(T), cudaMemcpyHostToDevice);
    
    const int threads = 256;
    const int blocks = batch_size;
    
    layernorm_forward_kernel_ori<<<blocks, threads