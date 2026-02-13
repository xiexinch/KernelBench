#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

template<typename T>
__global__ void l1_normalize_kernel_ori(
    const T* input,
    T* output,
    const int batch_size,
    const int dim,
    const float epsilon
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    // Each thread computes partial sum
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += stride) {
        thread_sum += fabsf(static_cast<float>(input[batch_idx * dim + i]));
    }
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Compute mean and normalize
    if (tid == 0) {
        float mean_val = sdata[0] / dim + epsilon;
        for (int i = 0; i < dim; i += stride) {
            int idx = batch_idx * dim + i;
            output[idx] = static_cast<T>(static_cast<float>(input[idx]) / mean_val);
        }
    }
}

template<typename T>
__global__ void l1_normalize_kernel_small_ori(
    const T* input,
    T* output,
    const int batch_size,
    const int dim,
    const float epsilon
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / dim;
    const int elem_idx = idx % dim;
    
    if (batch_idx < batch_size && elem_idx < dim) {
        // Compute mean for this batch using warp shuffle
        float thread_val = fabsf(static_cast<float>(input[idx]));
        
        // Warp reduction for mean
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_val += __shfl_down_sync(0xffffffff, thread_val, offset);
        }
        
        float mean_val = __shfl_sync(0xffffffff, thread_val, 0) / dim + epsilon;
        output[idx] = static_cast<T>(static_cast<float>(input[idx]) / mean_val);
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    // Extract parameters from the input shape
    const int batch_size = in_batch;
    const int dim = in_elems / batch_size;  // Assuming in_elems = batch_size * dim
    
    const float epsilon = 1e-8f;
    
    // Choose kernel based on dimension size
    if (dim <= 1024) {
        // Use warp-level reduction kernel for smaller dimensions
        const int threads_per_block = 256;
        const int total_elements = batch_size * dim;
        const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        l1_normalize_kernel_small_ori<T><<<num_blocks, threads_per_block, 0, stream>>>(
            input,
            output,
            batch_size,
            dim,
            epsilon
        );
    } else {
        // Use shared memory reduction kernel for larger dimensions
        const int threads_per_block = 256;
        const int shared_mem_size = threads_per_block * sizeof(float);
        
        l1_normalize_kernel_ori<T><<<batch_size, threads_per_block, shared_mem_size, stream>>>(
            input,
            output,
            batch_size,
            dim,
            epsilon
        );
    }
}