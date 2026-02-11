__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_avgpool_gelu_scale_max_kernel_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int out_features,
    int pool_kernel_size,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int pooled_size = out_features / pool_kernel_size;
    const float* batch_input = input + batch_idx * out_features;
    
    extern __shared__ float shared_mem[];
    
    float local_max = -FLT_MAX;
    
    // Each thread processes multiple pooled elements
    for (int pool_idx = threadIdx.x; pool_idx < pooled_size; pool_idx += blockDim.x) {
        // AvgPool
        float sum = 0.0f;
        int start_idx = pool_idx * pool_kernel_size;
        #pragma unroll
        for (int k = 0; k < pool_kernel_size; k++) {
            sum += batch_input[start_idx + k];
        }
        float avg = sum / pool_kernel_size;
        
        // GELU
        float gelu_val = gelu_activation(avg);
        
        // Scale
        float scaled = gelu_val * scale_factor;
        
        // Track max
        local_max = fmaxf(local_max, scaled);
    }
    
    // Store local max in shared memory
    shared_mem[threadIdx.x] = local_max;
    __syncthreads();
    
    // Reduce to find global max for this batch
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] = fmaxf(shared_mem[threadIdx.x], shared_mem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    // Write result
    if (threadIdx.x == 0) {
        output[batch_idx] = shared_mem[0];
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
    int out_features = in_height;
    int pool_kernel_size = in_channels;
    float scale_factor = static_cast<float>(in_width);
    
    int shared_mem_size = THREADS_PER_BLOCK * sizeof(float);
    
    fused_avgpool_gelu_scale_max_kernel_opt<<<batch_size, THREADS_PER_BLOCK, shared_mem_size, stream>>>(
        input,
        output,
        batch_size,
        out_features,
        pool_kernel_size,
        scale_factor
    );
}