__global__ void mean_reduction_kernel_ori(const float* input, float* output, 
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
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    // Assuming reduction over dimension 1 (dim1 = 4096)
    // Input shape: [batch_size, dim1, dim2] = [128, 4096, 4095]
    // Output shape: [batch_size, dim2] = [128, 4095]
    
    int outer_size = in_batch;  // 128
    int reduce_dim_size = in_height;  // 4096
    int inner_size = in_channels;  // 4095
    
    dim3 block_size(256, 1);
    dim3 grid_size(outer_size, (inner_size + block_size.y - 1) / block_size.y);
    
    mean_reduction_kernel_ori<<<grid_size, block_size, 0, stream>>>(
        input, 
        output,
        reduce_dim_size, 
        inner_size, 
        outer_size
    );
}