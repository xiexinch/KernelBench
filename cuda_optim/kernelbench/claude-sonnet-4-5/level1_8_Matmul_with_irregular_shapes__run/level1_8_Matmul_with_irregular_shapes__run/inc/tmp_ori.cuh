#define TILE_SIZE 32

__global__ void matmul_kernel_opt(const float* A, const float* B, float* C, 
                              int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from A
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
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
    // Interpret input as matrix A of shape (M, K)
    // and assume second matrix B is stored right after A in input buffer
    // Based on the original torch usage: A is (M, K), B is (K, N)
    // So total input size = M*K + K*N
    // But the provided interface only gives one input pointer.
    // To match the original kernel logic, we reinterpret:
    // - A starts at input
    // - B starts at input + M*K
    // - Output C is of size M*N

    int M = in_batch;      // Reuse in_batch as M
    int K = in_height;     // Reuse in_height as K
    int N = in_channels;   // Reuse in_channels as N

    const float* A = reinterpret_cast<const float*>(input);
    const float* B = reinterpret_cast<const float*>(input + M * K);
    float* C = reinterpret_cast<float*>(output);

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel_opt<<<grid_size, block_size, 0, stream>>>(A, B, C, M, K, N);
}