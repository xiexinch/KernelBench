__global__ void matvec_kernel_ori(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (row < M) {
        float sum = 0.0f;
        
        // Each thread processes multiple elements with stride
        for (int k = tid; k < K; k += block_size) {
            sum += A[row * K + k] * B[k];
        }
        
        // Shared memory for reduction
        __shared__ float shared_mem[256];
        shared_mem[tid] = sum;
        __syncthreads();
        
        // Reduction in shared memory
        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_mem[tid] += shared_mem[tid + stride];
            }
            __syncthreads();
        }
        
        // Write result
        if (tid == 0) {
            C[row] = shared_mem[0];
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
    int M = in_batch;
    int K = in_height;
    
    const int block_size = 256;
    const int num_blocks = M;
    
    matvec_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input, 
        input + M * K, 
        output, 
        M, 
        K
    );
}