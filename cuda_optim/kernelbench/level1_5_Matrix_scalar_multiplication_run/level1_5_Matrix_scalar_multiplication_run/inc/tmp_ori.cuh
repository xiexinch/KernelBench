__global__ void matrix_scalar_mul_kernel_opt(const float* __restrict__ A, float* __restrict__ out, float s, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Vectorized load/store using float4 for better memory throughput
    int vec_size = size / 4;
    if (idx < vec_size) {
        float4* A_vec = (float4*)A;
        float4* out_vec = (float4*)out;
        
        float4 val = A_vec[idx];
        val.x *= s;
        val.y *= s;
        val.z *= s;
        val.w *= s;
        out_vec[idx] = val;
    }
    
    // Handle remaining elements
    int remainder_idx = vec_size * 4 + (idx - vec_size);
    if (idx >= vec_size && remainder_idx < size) {
        out[remainder_idx] = A[remainder_idx] * s;
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
    int size = in_elems;
    float s = 3.14;
    const int block_size = 256;
    const int num_blocks = ((size / 4) + block_size - 1) / block_size;
    
    matrix_scalar_mul_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input, 
        output, 
        s, 
        size
    );
}