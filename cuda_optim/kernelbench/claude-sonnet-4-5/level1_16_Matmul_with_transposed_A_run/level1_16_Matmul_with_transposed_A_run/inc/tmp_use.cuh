#define TILE_SIZE 32

__global__ void matmul_transpose_kernel_opt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
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
        // Load tile from A^T (A has shape K x M, we want M x K)
        // A^T[row][t*TILE_SIZE + tx] = A[t*TILE_SIZE + tx][row]
        int a_k = t * TILE_SIZE + tx;
        int a_m = row;
        if (a_k < K && a_m < M) {
            As[ty][tx] = A[a_k * M + a_m];  // A is stored as (K, M)
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B (B has shape K x N)
        int b_k = t * TILE_SIZE + ty;
        int b_n = col;
        if (b_k < K && b_n < N) {
            Bs[ty][tx] = B[b_k * N + b_n];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
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
    const int K = in_batch;
    const int M = in_height;
    const int N = out_height;
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_transpose_kernel_opt<<<numBlocks, threadsPerBlock, 0, stream>>>(
        input,
        input + in_batch * in_height,
        output,
        M, K, N
    );
}