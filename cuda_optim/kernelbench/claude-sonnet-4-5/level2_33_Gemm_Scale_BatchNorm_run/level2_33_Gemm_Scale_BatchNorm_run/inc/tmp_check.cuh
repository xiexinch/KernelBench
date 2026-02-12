#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void gemm_scale_kernel_ori(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        sum += bias[col];
        sum *= scale[col];
        output[row * out_features + col] = sum;
    }
}

__global__ void compute_mean_var_kernel_ori(
    const float* __restrict__ input,
    float* __restrict__ mean,
    float* __restrict__ var,
    int batch_size,
    int features
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < features) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int i = 0; i < batch_size; i++) {
            float val = input[i * features + col];
            sum += val;
            sum_sq += val * val;
        }
        
        mean[col] = sum / batch_size;
        var[col] = sum_sq / batch_size - mean[col] * mean[col];
    }
}

__global__ void batch_norm_kernel_ori(
    float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int batch_size,
    int features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * features;
    
    if (idx < total) {
        int col = idx % features;
        float normalized = (input[idx] - mean[col]) / sqrtf(var[col] + eps);
        input[idx] = normalized * weight[col] + bias[col];
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
    float* input_f = reinterpret_cast<float*>(input);
    float* output_f = reinterpret_cast<float*>(output);
    
    int batch_size = in_batch;
    int in_features = in_channels * in_height * in_width;
    int out_features = out_channels * out_height * out_width;
    
    float *weight, *bias, *scale, *bn_weight, *bn_bias, *mean, *var;
    cudaMalloc(&weight, out_features * in_features * sizeof(float));
    cudaMalloc(&bias, out_features * sizeof(float));
    cudaMalloc(&scale, out_features * sizeof(float));
    cudaMalloc(&bn_weight, out_features * sizeof(float));
    cudaMalloc(&bn_bias, out_features * sizeof(float));
    cudaMalloc(&mean, out_features * sizeof(float));
    cudaMalloc(&var, out_features * sizeof(float));
    
    cudaMemset(weight, 0, out_features * in_features * sizeof(float));
    cudaMemset(bias, 0, out_features * sizeof(float));
    cudaMemset(scale, 0, out_features * sizeof(float));
    cudaMemset(bn_weight, 0, out_features * sizeof(float));
    cudaMemset(bn_bias, 0, out_features * sizeof(float));
    
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim((out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);
    
    gemm_scale_kernel_ori<<<grid_dim, block_dim, 0, stream>>>(
        input_f, weight, bias, scale, output_f,
        batch_size, in_features, out_features
    );
    
    int blocks_stat = (out_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_mean_var_kernel_ori<<<blocks_stat, BLOCK_SIZE, 0, stream>>>(
        output_f, mean, var, batch_size, out_features
    );
    
    float eps = 1e-5f;
    int total = batch_size * out_features;
    int blocks_bn = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    batch_norm_kernel_ori<<<blocks_bn, BLOCK_SIZE, 0, stream>>>(
        output_f, mean, var, bn_weight, bn_bias, eps, batch_size, out_features
    );
    
    cudaFree(weight);
    cudaFree(bias);
    cudaFree(scale);
    cudaFree(bn_weight);
    cudaFree(bn_bias);
    cudaFree(mean);
    cudaFree(var);
}