#include <cuda_runtime.h>

__global__ void fused_activations_kernel_opt(const float* input, const float* add_value, 
                                         float* output, int batch_size, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * features;
    
    if (idx < total_size) {
        int feature_idx = idx % features;
        float val = input[idx] + add_value[feature_idx];
        val = swish(val);
        val = tanh_activation(val);
        val = gelu(val);
        val = hardtanh(val, -1.0f, 1.0f);
        output[idx] = val;
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
    int features = in_elems / in_batch;
    
    const float* input_data = reinterpret_cast<const float*>(input);
    const float* add_value_data = reinterpret_cast<const float*>(input) + in_elems;
    float* output_data = reinterpret_cast<float*>(output);
    
    int total_size = batch_size * features;
    const int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    
    fused_activations_kernel_opt<<<blocks, threads, 0, stream>>>(
        input_data,
        add_value_data,
        output_data,
        batch_size,
        features
    );
}