__global__ void mean_reduction_kernel_opt(const float* input, float* output, 
                                      int reduce_dim_size, int inner_size, 
                                      int outer_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    float sum = 0.0f;
    
    // Each thread accumulates elements along the reduction dimension
    for (int i = threadIdx.x; i < reduce_dim_size; i += blockDim.x) {
        int idx = outer_idx * reduce_dim_size * inner_size + i * inner_size + inner_idx;
        sum += input[idx];
    }
    
    // Reduce within warp using shuffle operations
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Use shared memory for cross-warp reduction
    __shared__ float shared_sum[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    if (lane == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (threadIdx.x == 0) {
            int out_idx = outer_idx * inner_size + inner_idx;
            output[out_idx] = sum / reduce_dim_size;
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
    // Assume reduction is along dim=1 (in_channels), consistent with typical usage
    // This maps to: outer_size = in_batch, reduce_dim_size = in_channels, inner_size = in_width
    // Note: in_height is unused here assuming 3D tensor [batch, channels, width]
    int outer_size = in_batch;
    int reduce_dim_size = in_channels;
    int inner_size = in_width;

    // Adjust block and grid dimensions as in original
    dim3 block_size(256, 1);
    dim3 grid_size(outer_size, (inner_size + block_size.y - 1) / block_size.y);

    mean_reduction_kernel_opt<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        reduce_dim_size,
        inner_size,
        outer_size
    );
}