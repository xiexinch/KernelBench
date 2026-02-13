#include <cuda_runtime.h>
#include <float.h>
#include <math.h>




__global__ void softmax_kernel_ori(const float* __restrict__ x, float* __restrict__ out, int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float* x_row = x + row * dim;
    float* out_row = out + row * dim;
    
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    
    // Shared memory for warp-level reductions
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];
    
    // Find maximum value
    float thread_max = -FLT_MAX;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_max = fmaxf(thread_max, x_row[i]);
    }
    
    float warp_max = warp_reduce_max(thread_max);
    if (lane == 0) {
        shared_max[wid] = warp_max;
    }
    __syncthreads();
    
    // Final reduction across warps
    float row_max = -FLT_MAX;
    if (tid < 32) {
        row_max = (tid < (blockDim.x + 31) / 32) ? shared_max[tid] : -FLT_MAX;
        row_max = warp_reduce_max(row_max);
    }
    __syncthreads();
    if (tid == 0) {
        shared_max[0] = row_max;
    }
    __syncthreads();
    row_max = shared_max[0];
    
    // Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = expf(x_row[i] - row_max);
        out_row[i] = val;
        thread_sum += val;
    }
    
    float warp_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) {
        shared_sum[wid] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    float row_sum = 0.0f;
    if (tid < 32) {
        row_sum = (tid < (blockDim.x + 31) / 32) ? shared_sum[tid] : 0.0f;
        row_sum = warp_reduce_sum(row_sum);
    }
    __syncthreads();
    if (tid == 0) {
        shared_sum[0] = row_sum;
    }
    __syncthreads();
    row_sum = shared_sum[0];
    
    // Normalize
    for (int i = tid; i < dim; i += blockDim.x) {
        out_row[i] = out_row[i] / row_sum;
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
    int dim = in_elems / in_batch;

    const int threads = 256;
    const int blocks = batch_size;

    softmax_kernel_ori<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        batch_size,
        dim
    );
}