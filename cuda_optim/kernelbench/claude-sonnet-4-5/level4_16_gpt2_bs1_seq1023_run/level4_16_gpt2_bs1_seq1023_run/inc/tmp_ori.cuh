#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_gelu_kernel_opt(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void layernorm_kernel_opt(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int hidden_size,
    float eps
) {
    int idx = blockIdx.x;
    if (idx < batch_size) {
        const float* x = input + idx * hidden_size;
        float* y = output + idx * hidden_size;
        
        // Compute mean
        float sum = 0.0f;
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            sum += x[i];
        }
        
        __shared__ float shared_sum[256];
        shared_sum[threadIdx.x] = sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        float mean = shared_sum[0] / hidden_size;
        
        // Compute variance
        float var_sum = 0.0f;
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float diff = x[i] - mean;
            var_sum += diff * diff;
        }
        
        shared_sum[threadIdx.x] = var_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        float variance = shared_sum[0] / hidden_size;
        float inv_std = rsqrtf(variance + eps);
        
        // Normalize and apply affine transformation
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            y[i] = ((x[i] - mean) * inv_std) * gamma[i] + beta[i];
        }
    }
}

__global__ void softmax_kernel_opt(const float* input, float* output, int batch_size, int dim) {
    int idx = blockIdx.x;
    if (idx < batch_size) {
        const float* x = input + idx * dim;
        float* y = output + idx * dim;
        
        // Find max
        float max_val = -FLT_MAX;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            max_val = fmaxf(max_val, x[i]);
        }
        
        __shared__ float shared_max[256];
        shared_max[threadIdx.x] = max_val;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
            }
            __syncthreads();
        }
        
        max_val = shared_max[0];
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            float exp_val = expf(x[i] - max_val);
            y[i] = exp_val;
            sum += exp_val;
        }
        
        __shared__ float shared_sum[256];
        shared_sum[threadIdx.x] = sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        sum = shared_sum[0];
        
        // Normalize
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            y[i] = y[i] / sum;
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
    const int block_size = 256;
    int num_blocks = (in_elems + block_size - 1) / block_size;
    fused_gelu_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        in_elems
    );
}