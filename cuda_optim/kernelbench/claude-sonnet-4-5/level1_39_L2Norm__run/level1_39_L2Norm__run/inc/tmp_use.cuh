__global__ void l2_norm_kernel_opt(const float* __restrict__ x, float* __restrict__ out, 
                                int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_sum[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Compute partial sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += block_size) {
        float val = x[row * dim + i];
        local_sum += val * val;
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Broadcast the norm to all threads
    __shared__ float norm;
    if (tid == 0) {
        norm = sqrtf(shared_sum[0] + 1e-12f);
    }
    __syncthreads();
    
    // Normalize and write output
    for (int i = tid; i < dim; i += block_size) {
        out[row * dim + i] = x[row * dim + i] / norm;
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
    int batch_size = in_batch;
    int dim = in_elems / in_batch;

    const int threads = 256;
    const int blocks = batch_size;
    const int shared_mem = threads * sizeof(float);

    l2_norm_kernel_opt<<<blocks, threads, shared_mem, stream>>>(
        input,
        output,
        batch_size,
        dim
    );
}