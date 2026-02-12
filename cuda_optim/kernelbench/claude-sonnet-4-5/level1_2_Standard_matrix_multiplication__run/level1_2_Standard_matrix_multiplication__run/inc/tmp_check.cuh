#define TILE_SIZE 32

__global__ void matmul_kernel_ori(const float* A, const float* B, float* C, 
                              int M, int K, int N) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K) {
            shared_A[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            shared_A[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        if ((t * TILE_SIZE + ty) < K && col < N) {
            shared_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            shared_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
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
    int N = out_height;
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                    (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        input, 
        input + M * K, 
        output, 
        M, K, N
    );
}