#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_logsumexp_activations_kernel_opt(
    const float* input, 
    float* output, 
    int batch_size, 
    int features) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        const float* row = input + batch_idx * features;
        
        // LogSumExp computation
        float max_val = -FLT_MAX;
        for (int i = 0; i < features; i++) {
            max_val = fmaxf(max_val, row[i]);
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < features; i++) {
            sum_exp += expf(row[i] - max_val);
        }
        
        float logsumexp_val = max_val + logf(sum_exp);
        
        // First LeakyReLU
        float x = logsumexp_val;
        x = (x > 0.0f) ? x : 0.01f * x;
        
        // Second LeakyReLU
        x = (x > 0.0f) ? x : 0.01f * x;
        
        // First GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        float gelu_val = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
        
        // Second GELU
        x = gelu_val;
        x = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
        
        output[batch_idx] = x;
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
) {
    // Map 2D tensor dimensions: input [in_batch, in_width], output [out_batch, 1]
    // where in_width = features and in_batch = batch_size
    int batch_size = in_batch;
    int features = in_width;
    
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    
    fused_logsumexp_activations_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<float*>(output),
        batch_size,
        features
    );
}