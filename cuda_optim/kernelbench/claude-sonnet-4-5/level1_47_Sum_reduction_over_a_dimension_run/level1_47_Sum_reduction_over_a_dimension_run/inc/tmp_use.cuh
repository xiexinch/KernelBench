__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void sum_reduction_dim1_kernel_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int dim1,
    int dim2
) {
    int batch_idx = blockIdx.x;
    int dim2_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || dim2_idx >= dim2) return;
    
    float sum = 0.0f;
    
    // Each thread reduces multiple elements if needed
    for (int i = threadIdx.x; i < dim1; i += blockDim.x) {
        int input_idx = batch_idx * dim1 * dim2 + i * dim2 + dim2_idx;
        sum += input[input_idx];
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // Shared memory for warp results
    __shared__ float warp_sums[32];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? warp_sums[threadIdx.x] : 0.0f;
        sum = warp_reduce_sum(sum);
        
        if (threadIdx.x == 0) {
            int output_idx = batch_idx * dim2 + dim2_idx;
            output[output_idx] = sum;
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
    int batch_size = in_batch;
    int dim1 = in_height;
    int dim2 = in_channels;
    
    dim3 grid(batch_size, dim2);
    int block_size = min(1024, ((dim1 + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);
    
    sum_reduction_dim1_kernel_opt<<<grid, block_size, 0, stream>>>(
        input,
        output,
        batch_size,
        dim1,
        dim2
    );
}