#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

template <typename scalar_t>
__global__ void softmax_forward_kernel_ori(
    const scalar_t* input,
    scalar_t* output,
    const int batch_size,
    const int num_features,
    const int stride) {
    
    extern __shared__ float shared_mem[];
    float* max_vals = shared_mem;
    float* sum_vals = &shared_mem[blockDim.x];
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Find maximum value in the row (online)
    float thread_max = -FLT_MAX;
    for (int i = tid; i < num_features; i += blockDim.x) {
        float val = static_cast<float>(input[batch_idx * stride + i]);
        thread_max = fmaxf(thread_max, val);
    }
    
    // Reduce to find global max
    max_vals[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + s]);
        }
        __syncthreads();
    }
    
    const float row_max = max_vals[0];
    __syncthreads();
    
    // Compute exponentials and sum (online)
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        float val = static_cast<float>(input[batch_idx * stride + i]);
        float exp_val = expf(val - row_max);
        output[batch_idx * stride + i] = static_cast<scalar_t>(exp_val);
        thread_sum += exp_val;
    }
    
    // Reduce to find global sum
    sum_vals[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_vals[tid] += sum_vals[tid + s];
        }
        __syncthreads();
    }
    
    const float row_sum = sum_vals[0];
    const float inv_sum = 1.0f / row_sum;
    
    // Normalize by sum
    for (int i = tid; i < num_features; i += blockDim.x) {
        float val = static_cast<float>(output[batch_idx * stride + i]);
        output[batch_idx * stride + i] = static_cast<scalar_t>(val * inv_sum);
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
    // Extract parameters from the input dimensions
    const int batch_size = in_batch;
    const int num_features = in_channels;  // Assuming channels dimension is the feature dimension
    const int stride = in_channels;  // Assuming contiguous layout
    
    // Optimize thread block size based on problem dimensions
    const int threads_per_block = 256;
    const int shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    dim3 grid(batch_size);
    dim3 block(threads_per_block);
    
    softmax_forward_kernel_ori<T><<<grid, block, shared_mem_size, stream>>>(
        input,
        output,
        batch_size,
        num_features,
        stride);
}