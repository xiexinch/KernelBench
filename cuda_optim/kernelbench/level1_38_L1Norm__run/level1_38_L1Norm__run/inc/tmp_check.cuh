__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void l1_norm_kernel_ori(const float* __restrict__ x, 
                                float* __restrict__ out, 
                                int batch_size, 
                                int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float* x_row = x + row * dim;
    float* out_row = out + row * dim;
    
    // Phase 1: Compute sum of absolute values using all threads
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += fabsf(x_row[i]);
    }
    
    // Reduce within warp
    local_sum = warp_reduce_sum(local_sum);
    
    // Shared memory for block-level reduction
    __shared__ float warp_sums[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        local_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
    }
    
    // Broadcast the mean to all threads
    __shared__ float mean_val;
    if (threadIdx.x == 0) {
        mean_val = local_sum / dim;
    }
    __syncthreads();
    
    // Phase 2: Divide by mean
    float mean_inv = 1.0f / mean_val;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = x_row[i] * mean_inv;
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
    
    l1_norm_kernel_ori<<<blocks, threads, 0, stream>>>(
        input,
        output,
        batch_size,
        dim
    );
}