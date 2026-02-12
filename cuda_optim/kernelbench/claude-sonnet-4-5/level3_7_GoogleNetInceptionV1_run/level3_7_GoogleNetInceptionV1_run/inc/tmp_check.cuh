__global__ void relu_kernel_ori(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] > 0.0f ? data[idx] : 0.0f;
    }
}

__global__ void concat_kernel_4_ori(
    const float* input1, const float* input2, const float* input3, const float* input4,
    float* output,
    int batch_size, int height, int width,
    int c1, int c2, int c3, int c4) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * height * width * (c1 + c2 + c3 + c4);
    
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % (c1 + c2 + c3 + c4);
        int b = idx / (width * height * (c1 + c2 + c3 + c4));
        
        int hw = h * width + w;
        int base_offset = b * height * width;
        
        if (c < c1) {
            output[idx] = input1[base_offset * c1 + c * height * width + hw];
        } else if (c < c1 + c2) {
            int c_offset = c - c1;
            output[idx] = input2[base_offset * c2 + c_offset * height * width + hw];
        } else if (c < c1 + c2 + c3) {
            int c_offset = c - c1 - c2;
            output[idx] = input3[base_offset * c3 + c_offset * height * width + hw];
        } else {
            int c_offset = c - c1 - c2 - c3;
            output[idx] = input4[base_offset * c4 + c_offset * height * width + hw];
        }
    }
}

__global__ void avgpool_flatten_kernel_ori(const float* input, float* output, 
                                       int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels;
    
    if (idx < total_outputs) {
        int c = idx % channels;
        int b = idx / channels;
        
        float sum = 0.0f;
        int input_offset = b * channels * height * width + c * height * width;
        
        for (int i = 0; i < height * width; i++) {
            sum += input[input_offset + i];
        }
        
        output[idx] = sum / (height * width);
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
    int size = in_elems;
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    relu_kernel_ori<<<num_blocks, block_size>>>(input, size);
}