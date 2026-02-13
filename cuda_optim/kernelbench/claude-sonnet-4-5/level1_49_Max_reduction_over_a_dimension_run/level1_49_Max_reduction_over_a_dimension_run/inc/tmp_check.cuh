#include <cuda_runtime.h>
#include <float.h>
#include <algorithm>




__global__ void max_reduction_kernel_ori(const float* input, float* output, 
                                     int batch_size, int reduce_dim, int inner_dim) {
    int batch_idx = blockIdx.y;
    int inner_idx = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || inner_idx >= inner_dim) return;
    
    float max_val = -FLT_MAX;
    
    // Each thread processes elements along the reduction dimension
    for (int i = threadIdx.x; i < reduce_dim; i += blockDim.x) {
        int input_idx = batch_idx * reduce_dim * inner_dim + i * inner_dim + inner_idx;
        max_val = fmaxf(max_val, input[input_idx]);
    }
    
    // Warp-level reduction
    max_val = warp_reduce_max(max_val);
    
    // Shared memory for block-level reduction
    __shared__ float shared[32][33]; // 33 to avoid bank conflicts
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    if (lane == 0) {
        shared[wid][threadIdx.y] = max_val;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (threadIdx.x < 32) {
        max_val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[threadIdx.x][threadIdx.y] : -FLT_MAX;
        max_val = warp_reduce_max(max_val);
        
        if (threadIdx.x == 0) {
            output[batch_idx * inner_dim + inner_idx] = max_val;
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
    // Reconstruct dimensions as in the original torch code
    // The original kernel reduces over a specific dimension.
    // Based on the tensor layout and typical usage, we assume:
    // - Input shape: [in_batch, in_height, in_channels, in_width]
    // - Reduction is performed over the 'in_channels' dimension (dim=2)
    // - Thus: batch_size = in_batch * in_height
    //         reduce_dim = in_channels
    //         inner_dim = in_width

    int batch_size = in_batch * in_height;
    int reduce_dim = in_channels;
    int inner_dim = in_width;

    dim3 block(256, 1);
    if (inner_dim > 1) {
        block.x = 256;
        block.y = std::min(4, inner_dim);
    }

    dim3 grid((inner_dim + block.y - 1) / block.y, batch_size);

    max_reduction_kernel_ori<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        batch_size,
        reduce_dim,
        inner_dim
    );
}