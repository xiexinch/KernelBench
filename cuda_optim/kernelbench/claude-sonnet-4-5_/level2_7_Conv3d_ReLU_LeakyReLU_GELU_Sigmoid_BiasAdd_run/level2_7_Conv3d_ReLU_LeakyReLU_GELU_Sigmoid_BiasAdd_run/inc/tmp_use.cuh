#include <cuda_runtime.h>
#include <math.h>

__device__ float gelu_activation(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(tanh_arg));
}

__global__ void fused_activations_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * depth * height * width;
    
    if (idx < total_size) {
        int spatial_size = depth * height * width;
        int temp = idx / spatial_size;
        int channel_idx = temp % channels;
        
        float val = input[idx];
        
        val = fmaxf(val, 0.0f);
        
        val = (val > 0.0f) ? val : (0.01f * val);
        
        val = gelu_activation(val);
        
        val = 1.0f / (1.0f + expf(-val));
        
        val = val + bias[channel_idx];
        
        output[idx] = val;
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
    float* input_f = reinterpret_cast<float*>(input);
    float* output_f = reinterpret_cast<float*>(output);
    
    float* bias_d = nullptr;
    cudaMalloc(&bias_d, in_channels * sizeof(float));
    cudaMemset(bias_d, 0, in_channels * sizeof(float));
    
    const int block_size = 256;
    const int num_blocks = (in_elems + block_size - 1) / block_size;
    
    fused_activations_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input_f,
        bias_d,
        output_f,
        in_batch,
        in_channels,
        1,
        in_height,
        in_width
    );
    
    cudaFree(bias_d);
}