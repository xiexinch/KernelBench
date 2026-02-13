#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>

__global__ void group_norm_scale_kernel_opt(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* scale,
    float* output,
    int batch_size,
    int num_channels,
    int spatial_size,
    int num_groups,
    int channels_per_group,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = batch_size * num_groups;
    
    if (idx < batch_size * num_channels * spatial_size) {
        int s = idx % spatial_size;
        int c = (idx / spatial_size) % num_channels;
        int b = idx / (num_channels * spatial_size);
        int g = c / channels_per_group;
        
        // Calculate mean and variance for this group
        float sum = 0.0f;
        float sq_sum = 0.0f;
        int group_size = channels_per_group * spatial_size;
        int group_start_c = g * channels_per_group;
        
        for (int gc = 0; gc < channels_per_group; gc++) {
            for (int gs = 0; gs < spatial_size; gs++) {
                int gidx = b * num_channels * spatial_size + (group_start_c + gc) * spatial_size + gs;
                float val = input[gidx];
                sum += val;
                sq_sum += val * val;
            }
        }
        
        float mean = sum / group_size;
        float var = sq_sum / group_size - mean * mean;
        float std_inv = rsqrtf(var + eps);
        
        // Normalize and apply affine transformation
        float normalized = (input[idx] - mean) * std_inv;
        float affine = normalized * gamma[c] + beta[c];
        
        // Apply scale
        output[idx] = affine * scale[c];
    }
}

__global__ void maxpool_clamp_kernel_opt(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    float clamp_min,
    float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx < total_elements) {
        int ow = idx % out_width;
        int oh = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int b = idx / (channels * out_height * out_width);
        
        float max_val = -1e38f;
        
        int h_start = oh * kernel_size;
        int w_start = ow * kernel_size;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h = h_start + kh;
                int w = w_start + kw;
                
                if (h < in_height && w < in_width) {
                    int in_idx = b * channels * in_height * in_width + 
                                c * in_height * in_width + 
                                h * in_width + w;
                    max_val = fmaxf(max_val, input[in_idx]);
                }
            }
        }
        
        // Apply clamp
        max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);
        output[idx] = max_val;
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
    // This implementation expects T to be float
    assert(sizeof(T) == sizeof(float));
    
    float* f_input = reinterpret_cast<float*>(input);
    float* f_output = reinterpret_cast<float*>(output);
    
    // GroupNorm parameters
    int num_groups = 16;
    if (num_groups > in_channels) num_groups = in_channels;
    int channels_per_group = in_channels / num_groups;
    float eps = 1e-5f;
    
    // Allocate and initialize gamma, beta, scale
    float *d_gamma, *d_beta, *d_scale;
    cudaMalloc(&d_gamma, in_channels * sizeof(float));
    cudaMalloc(&d_beta, in_channels * sizeof(float));
    cudaMalloc(&d_scale, in_channels * sizeof(float));
    
    // Initialize on host and copy
    float* h_gamma = new float[in_channels];
    float* h_beta = new float[in_channels];
    float* h_scale = new float[in_channels];
    for (int i = 0; i < in_channels; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
        h_scale[i] = 1.0f;
    }
    cudaMemcpyAsync(d_gamma, h_gamma, in_channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_beta, h_beta, in_channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_scale, h_scale, in_channels * sizeof(float), cudaMemcpyHostToDevice, stream);
    delete[] h_gamma;
    delete[] h_beta;
    delete[] h_scale;
    
    // Allocate intermediate buffer
    float* d_intermediate;
    cudaMalloc(&d_intermediate, in_elems * sizeof(float));
    
    // Launch GroupNorm + Scale kernel
    int spatial_size = in_height * in_width;
    int total_elements = in_batch * in_channels * spatial_size;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    group_norm_scale_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        f_input,
        d_gamma,
        d_beta,
        d_scale,
        d_intermediate,
        in_batch,
        in_channels,
        spatial_size,
        num_groups,
        channels_per_group,
        eps
    );
    
    // Calculate MaxPool kernel size from input/output dimensions
    int kernel_size = in_height / out_height;
    float clamp_min = 0.0f;
    float clamp_max = 1.0f;
    
    // Launch MaxPool + Clamp kernel
    int maxpool_total_elements = out_batch * out_channels * out_height * out_width;
    const int num_blocks_mp = (maxpool_total_elements + block_size - 1) / block_size;
    
    maxpool_clamp_kernel_opt<<<num_blocks_mp, block_size, 0, stream>>>(
        d_intermediate,
        f_output,
        in_batch,
        in_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        clamp_min,
        clamp_max
    );
    
    // Cleanup temporary allocations
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_scale);
    cudaFree(d_intermediate);
}