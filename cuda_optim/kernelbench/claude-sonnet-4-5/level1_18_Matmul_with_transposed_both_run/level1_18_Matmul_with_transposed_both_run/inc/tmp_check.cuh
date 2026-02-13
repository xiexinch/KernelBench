#define TILE_SIZE 32

__global__ void matmul_kernel_ori(const float* A, const float* B, float* C, 
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
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((t * TILE_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
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
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    // Interpret input as two matrices: A (M x K) and B (K x N)
    // According to the original code:
    //   A is (M, K) -> in_batch = M, in_height = K (assuming in_channels=1, in_width=1)
    //   B is (K, N) -> out_batch unused, out_height = K, out_width = N
    // But the kernel expects M, K, N directly.
    // From get_inputs():
    //   A shape: (4096*2, 1024*2) -> M = 8192, K = 2048
    //   B shape: (2048*2, 4096*2) -> but passed as B_T: (4096*2, 2048*2) -> K = 2048, N = 8192
    // So we deduce:
    int M = in_batch;
    int K = in_height;
    int N = out_width;

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel_ori<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(input + in_elems), // B starts after A
        reinterpret_cast<float*>(output),
        M, K, N
    );
}