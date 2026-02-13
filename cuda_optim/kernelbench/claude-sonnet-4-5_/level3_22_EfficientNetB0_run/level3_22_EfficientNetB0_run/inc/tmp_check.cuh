__global__ void batchnorm_relu6_kernel_ori(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ out,
    int N, int C, int HW,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    
    if (idx < total) {
        int c = (idx / HW) % C;
        float val = x[idx];
        float mean = running_mean[c];
        float var = running_var[c];
        float w = weight[c];
        float b = bias[c];
        
        float normalized = (val - mean) / sqrtf(var + eps);
        float result = normalized * w + b;
        result = fminf(fmaxf(result, 0.0f), 6.0f);
        out[idx] = result;
    }
}

__global__ void batchnorm_add_kernel_ori(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ out,
    int N, int C, int HW,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    
    if (idx < total) {
        int c = (idx / HW) % C;
        float val = x[idx];
        float mean = running_mean[c];
        float var = running_var[c];
        float w = weight[c];
        float b = bias[c];
        
        float normalized = (val - mean) / sqrtf(var + eps);
        float result = normalized * w + b + residual[idx];
        out[idx] = result;
    }
}

__global__ void batchnorm_relu_kernel_ori(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ out,
    int N, int C, int HW,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    
    if (idx < total) {
        int c = (idx / HW) % C;
        float val = x[idx];
        float mean = running_mean[c];
        float var = running_var[c];
        float w = weight[c];
        float b = bias[c];
        
        float normalized = (val - mean) / sqrtf(var + eps);
        float result = normalized * w + b;
        result = fmaxf(result, 0.0f);
        out[idx] = result;
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    int N = in_batch;
    int C = in_channels;
    int HW = in_height * in_width;
    float eps = 1e-5;
    
    const int threads = 256;
    const int blocks = (N * C * HW + threads - 1) / threads;
    
    // This is a placeholder entry that calls batchnorm_relu_kernel_ori
    // In actual usage, weight, bias, running_mean, running_var would be