#define TILE_SIZE 32
#define WARP_SIZE 32

__global__ void matmul_small_k_kernel_ori(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Each block computes a TILE_SIZE x TILE_SIZE tile of C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    __shared__ float As[TILE_SIZE][64];  // Maximum K=64
    __shared__ float Bs[64][TILE_SIZE];  // Maximum K=64
    
    float sum = 0.0f;
    
    // Load A and B into shared memory
    // Each thread loads elements
    for (int k = threadIdx.x; k < K; k += TILE_SIZE) {
        if (row < M) {
            As[threadIdx.y][k] = A[row * K + k];
        }
    }
    
    for (int k = threadIdx.y; k < K; k += TILE_SIZE) {
        if (col < N) {
            Bs[k][threadIdx.x] = B[k * N + col];
        }
    }
    
    __syncthreads();
    
    // Compute the dot product
    if (row < M && col < N) {
        #pragma unroll
        for (int k = 0; k < K; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
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
    int N = in_channels;
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_small_k_kernel_ori<<<grid, block, 0, stream>>>(
        input,
        output,
        output,
        M, N, K
    );
}