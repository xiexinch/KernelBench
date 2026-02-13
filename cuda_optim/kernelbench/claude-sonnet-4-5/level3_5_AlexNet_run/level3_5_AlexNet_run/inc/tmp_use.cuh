__global__ void relu_kernel_opt(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] > 0.0f ? data[idx] : 0.0f;
    }
}

__global__ void relu_bias_kernel_opt(const float* input, const float* bias, float* output, 
                                  int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    
    if (idx < total) {
        int feat_idx = idx % out_features;
        float val = input[idx] + bias[feat_idx];
        output[idx] = val > 0.0f ? val : 0.0f;
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
    // Determine which kernel to launch based on the presence of bias
    // Since we only have input and output pointers, and no bias pointer,
    // we assume this is the inplace ReLU case (relu_kernel_opt).
    // The relu_bias_kernel_opt requires a separate bias array, which isn't provided here.
    // So we map to relu_kernel_opt using output as the data array (in-place style).

    int size = out_elems; // assuming output is what we apply ReLU to
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel_opt<<<num_blocks, block_size, 0, stream>>>(reinterpret_cast<float*>(output), size);
}