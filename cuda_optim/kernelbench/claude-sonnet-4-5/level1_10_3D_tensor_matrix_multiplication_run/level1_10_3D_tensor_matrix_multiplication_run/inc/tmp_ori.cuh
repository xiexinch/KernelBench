#define TILE_SIZE 16

__global__ void batched_matmul_kernel_opt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N, int M, int K, int L
) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[batch_idx * M * K + row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (b_row < K && col < L) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * L + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < L) {
        C[batch_idx * M * L + row * L + col] = sum;
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
    int N = in_batch;
    int M = in_height;
    int K = in_channels;
    int L = out_width;
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((L + TILE_SIZE - 1) / TILE_SIZE, 
                (M + TILE_SIZE - 1) / TILE_SIZE,
                N);
    
    batched_matmul_kernel_opt<<<blocks, threads, 0, stream>>>(
        input,
        input + in_elems,
        output,
        N, M, K, L
    );
}