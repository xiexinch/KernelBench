#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "common.h"
#include "tmp_check.cuh"
#include "tmp_use.cuh"
struct ResultStruct {
    float ori_time;
    float opt_time;
    float time_rate;
    bool result_is_right;
};

template <typename T>
ResultStruct test_tmp(std::vector<int> input_size, std::vector<int> output_size) {
    ResultStruct result;
    CUDA_INIT();
    int in_batch = input_size[0];
    int in_height = input_size[1];
    int in_channels = input_size[2];
    int in_width = input_size[3];
    int out_batch = output_size[0];
    int out_height = output_size[1];
    int out_channels = output_size[2];
    int out_width = output_size[3];
    int in_elems = in_batch * in_height * in_channels * in_width;
    int out_elems = out_batch * out_height * out_channels * out_width;

    T *input = nullptr;
    T *output = nullptr;
    T *output_target = nullptr;
    T *input_cpu = nullptr;
    T *output_cpu = nullptr;
    T *output_target_cpu = nullptr;

    output_cpu = (T *)malloc(sizeof(T) * out_elems);
    output_target_cpu = (T *)malloc(sizeof(T) * out_elems);
    input_cpu = (T *)malloc(sizeof(T) * in_elems);
    cudaMalloc((void **)&input, sizeof(T) * in_elems);
    cudaMalloc((void **)&output, sizeof(T) * out_elems);
    cudaMalloc((void **)&output_target, sizeof(T) * out_elems);
    memset(output_cpu, 1, sizeof(T) * out_elems);
    memset(output_target_cpu, 2, sizeof(T) * out_elems);
    cudaMemset(output, 1, sizeof(T) * out_elems);
    cudaMemset(output_target, 2, sizeof(T) * out_elems);

    for(int i = 0; i < in_elems; i++) { input_cpu[i] = (i*7)%127; }
    cudaMemcpy(input, input_cpu, sizeof(T) * in_elems, cudaMemcpyHostToDevice);

    float total_time = 0.0;
    int test_count = 1000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 5; i++)
        test_tmp_kernel_ori(input, output, in_batch, in_height, in_channels, in_width, out_batch, out_height, out_channels, out_width, in_elems, out_elems, stream);
    for (int i = 0; i < test_count; i++) {
        cudaEventRecord(start, 0);
        test_tmp_kernel_ori(input, output, in_batch, in_height, in_channels, in_width, out_batch, out_height, out_channels, out_width, in_elems, out_elems, stream);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t;
        cudaEventElapsedTime(&t, start, stop);
        total_time += t;
    }
    result.ori_time = total_time / test_count;
    total_time = 0;
    for (int i = 0; i < test_count; i++) {
        cudaEventRecord(start, 0);
        test_tmp_kernel_opt(input, output_target, in_batch, in_height, in_channels, in_width, out_batch, out_height, out_channels, out_width, in_elems, out_elems, stream);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t;
        cudaEventElapsedTime(&t, start, stop);
        total_time += t;
    }
    result.opt_time = total_time / test_count;
    cudaDeviceSynchronize();
    cudaMemcpy(output_cpu, output, sizeof(T) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_target_cpu, output_target, sizeof(T) * out_elems, cudaMemcpyDeviceToHost);
    result.time_rate = result.opt_time / result.ori_time;
    result.result_is_right = checkresult<T>(output_cpu, output_target_cpu, out_elems);

    free(input_cpu);
    free(output_cpu);
    free(output_target_cpu);
    cudaFree(input);
    cudaFree(output);
    cudaFree(output_target);
    return result;
}

int main() {
    ResultStruct result1 = test_tmp<float>({1024,16384,1,1}, {1024,16384,1,1});
    printf("%lf ", result1.ori_time);
    printf("<time_before_opt>%f ms</time_before_opt>\n", result1.ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", result1.opt_time);
    printf("<runtime_ratio>%f</runtime_ratio>\n", result1.opt_time / result1.ori_time);
    printf("<precision>%s</precision>\n", result1.result_is_right ? "True" : "False");
}
