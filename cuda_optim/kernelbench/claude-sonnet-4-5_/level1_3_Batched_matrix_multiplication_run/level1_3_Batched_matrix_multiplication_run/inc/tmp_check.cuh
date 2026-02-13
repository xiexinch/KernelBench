#include <cuda_runtime.h>
#include <cstdint>

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

__global__ void bmm_kernel_ori(
    const float* A, 
    const float* B, 
    float* C,
    int batch_size,
    int m, int n, int k) {
    
    // Shared memory for tile of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int batch = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory
        int A_row = row;
        int A_col = t * TILE_SIZE + threadIdx.x;
        if (A_row < m && A_col < k) {
            As[threadIdx.y][threadIdx.x] = A[batch * m * k + A_row * k + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int B_row = t * TILE_SIZE + threadIdx.y;
        int B_col = col;
        if (B_row < k && B_col < n) {
            Bs[threadIdx.y][threadIdx.x] = B[batch * k * n + B_row * n + B_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result to C
    if (row < m && col < n) {
        C[batch * m * n + row * n + col] = sum;
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    // Extract parameters from input/output shapes
    // For BMM: input = A, output = C, B is the second input
    // We'll use the first half of input as A, second half as B
    int batch_size = in_batch;
    int m = in_height;
    int k = in_channels;  // Using in_channels as k dimension
    int n = out_width;    // Using out_width as n dimension
    
    // Configure grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (n + TILE_SIZE - 1) / TILE_SIZE,
        (m + TILE_SIZE - 1) / TILE_SIZE,
        batch_size
    );
    
    // Split input into A and B
    T* A = input;
    T* B = input + batch_size * m * k;  // B starts after A
    T* C = output;
    
    // Launch kernel
    bmm_kernel_ori<<<gridDim, blockDim, 0, stream>>>(
        A,
        B,
        C,
        batch_size,
        m, n, k
    );
}