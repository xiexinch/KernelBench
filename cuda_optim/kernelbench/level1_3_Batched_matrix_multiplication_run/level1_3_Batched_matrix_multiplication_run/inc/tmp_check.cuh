__global__ void batched_matmul_kernel_ori(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size, int m, int k, int n) {
    
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    int A_offset = batch_idx * m * k;
    int B_offset = batch_idx * k * n;
    int C_offset = batch_idx * m * n;
    
    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int tile_col = tile * TILE_SIZE + threadIdx.x;
        int tile_row = tile * TILE_SIZE + threadIdx.y;
        
        if (row < m && tile_col < k) {
            tile_A[threadIdx.y][threadIdx.x] = A[A_offset + row * k + tile_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (tile_row < k && col < n) {
            tile_B[threadIdx.y][threadIdx.x] = B[B_offset + tile_row * n + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[C_offset + row * n + col] = sum;
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
    int batch_size = in_batch;
    int m = in_height;
    int k = in_channels;
    int n = out_width;
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, 
              (m + TILE_SIZE - 1) / TILE_SIZE,
              batch_size);
    
    batched_matmul_kernel_ori<<<grid, block, 0, stream>>>(
        input,
        output,
        output,
        batch_size, m, k, n
    );
}