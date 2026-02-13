#include <cuda_runtime.h>
#include <math.h>

__global__ void add_bias_sigmoid_kernel_ori(const float* input, const float* bias, 
                                         float* output, int batch_size, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * hidden_size;
    
    if (idx < total_size) {
        int col = idx % hidden_size;
        float val = input[idx] + bias[col];
        output[idx] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void sum_reduction_kernel_ori(const float* input, float* output, 
                                      int batch_size, int hidden_size) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            sum += input[batch_idx * hidden_size + i];
        }
        
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Block-level reduction using shared memory
        __shared__ float shared_sum[32];
        int lane = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;
        
        if (lane == 0) {
            shared_sum[warp_id] = sum;
        }
        __syncthreads();
        
        if (warp_id == 0) {
            sum = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_sum[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (threadIdx.x == 0) {
                output[batch_idx] = sum;
            }
        }
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
    // Cast to float pointers as the kernels operate on float data
    float* input_f = reinterpret_cast<float*>(input);
    float* output_f = reinterpret_cast<float*>(output);
    
    // Map to sum_reduction_kernel_ori parameters
    // Assuming 2D input: [batch_size, hidden_size]
    int batch_size = in_batch;
    int hidden_size = in_elems / in_batch;
    if (hidden_size < 1) hidden_size = 1;
    
    const int threads = 256;
    // Launch with batch_size blocks, each reducing one row
    sum_reduction_kernel_ori<<<batch_size, threads, 0, stream>>>(
        input_f, output_f, batch_size, hidden_size
    );
}