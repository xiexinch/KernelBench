#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

__global__ void fused_ops_kernel_opt(
    const float* input,
    const float* scaling_factor,
    const float* bias,
    float* output,
    int size,
    int out_channels,
    int spatial_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int channel = (idx / spatial_size) % out_channels;
        float val = input[idx];
        val = val * scaling_factor[channel];
        val = tanhf(val);
        val = val * bias[channel];
        val = 1.0f / (1.0f + expf(-val));
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
    static_assert(std::is_same<T, float>::value, "This kernel only supports float type");
    
    const float* input_f = static_cast<const float*>(input);
    float* output_f = static_cast<float*>(output);
    
    // Map 4D dimensions to original 5D logic:
    // Original expects: batch, channels, depth, height, width
    // We treat: in_batch -> batch, in_channels -> channels, 
    // in_height * in_width -> spatial_size (depth * height * width)
    int batch = in_batch;
    int channels = in_channels;
    int spatial_size = in_height * in_width;
    int size = in_elems;
    
    // Allocate temporary scaling_factor and bias arrays
    // Since the signature doesn't provide them, we allocate and initialize to 1.0f
    float *d_scaling_factor, *d_bias;
    cudaMalloc(&d_scaling_factor, channels * sizeof(float));
    cudaMalloc(&d_bias, channels * sizeof(float));
    
    // Initialize with 1.0f using host-side fill and async copy
    float* h_scaling_factor = new float[channels];
    float* h_bias = new float[channels];
    for (int i = 0; i < channels; ++i) {
        h_scaling_factor[i] = 1.0f;
        h_bias[i] = 1.0f;
    }
    cudaMemcpyAsync(d_scaling_factor, h_scaling_factor, channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bias, h_bias, channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    delete[] h_scaling_factor;
    delete[] h_bias;
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_ops_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        input_f,
        d_scaling_factor,
        d_bias,
        output_f,
        size,
        channels,
        spatial_size
    );
    
    // Cleanup
    cudaFree(d_scaling_factor);
    cudaFree(d_bias);
}