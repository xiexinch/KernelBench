#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "common.h"
#include <float.h>
#include "tmp_check.cuh" 
#include "tmp_use.cuh" 
struct ResultStruct {
    float ori_time;
    float opt_time;
    float time_rate;
    bool result_is_right;
  };
  __global__ void warm_up()
{

}

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
    int weight_len = in_height * in_width * in_channels;
    T *input = nullptr;
    T *output = nullptr;
    T *output_target = nullptr;
    T *input_cpu = nullptr;
    T *output_cpu = nullptr;
    T *output_target_cpu = nullptr;
    T *weight_cpu = nullptr;
    T *bias_cpu = nullptr;
    T *weight = nullptr;
    T *bias = nullptr;


    // 分配 CPU 内存
    input_cpu = (T *)malloc(sizeof(T) * in_elems);
    output_cpu = (T *)malloc(sizeof(T) * out_elems);
    output_target_cpu = (T *)malloc(sizeof(T) * out_elems);
    weight_cpu = (T *)malloc(sizeof(T) * weight_len);
    bias_cpu = (T *)malloc(sizeof(T) * weight_len);

    // 分配 GPU 内存
    cudaMalloc((void **)&input, sizeof(T) * in_elems);
    cudaMalloc((void **)&output, sizeof(T) * out_elems);
        cudaMalloc((void **)&output_target, sizeof(T) * out_elems);
    memset(output_cpu, 1, sizeof(T) * out_elems); 
    memset(output_target_cpu, 2, sizeof(T) * out_elems); 
    cudaMemset (output,1, sizeof(T) * out_elems);
    cudaMemset (output_target,2, sizeof(T) * out_elems);
    cudaMalloc((void **)&weight, sizeof(T) * weight_len);
    cudaMalloc((void **)&bias, sizeof(T) * weight_len);

    //input初始化
    for(int i = 0; i < in_batch * in_height * in_width * in_channels; i++)
    {
        input_cpu[i] = (i*7)%127;
    }
    for(int i = 0; i < weight_len; i++)
    {
        weight_cpu[i] = 1.0;
        bias_cpu[i] = 0.0;
    }
    // 将数据从 CPU 拷贝到 GPU
    cudaMemcpy(input, input_cpu, sizeof(T) * in_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(weight, weight_cpu, sizeof(T) * weight_len, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, bias_cpu, sizeof(T) * weight_len, cudaMemcpyHostToDevice);


        //###########ori#############
        float total_time = 0.0;
        int test_count = 1000;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        int warm_up_count = 5;
        for (int i = 0 ; i < warm_up_count; i++)//warm up
        {
            test_tmp_kernel_ori(input, output, weight,bias,in_batch, in_height, in_channels, in_width,out_batch ,out_height ,out_channels, out_width,in_elems,out_elems,stream);
        }
        for (int i = 0; i < test_count; i++)
        {
            cudaEventRecord(start, 0);
            //优化前
            test_tmp_kernel_ori(input, output, weight,bias, in_batch, in_height, in_channels, in_width,out_batch ,out_height ,out_channels, out_width,in_elems,out_elems,stream);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float elapsed_time = 0.0f;
            cudaEventElapsedTime(&elapsed_time, start, stop);
            total_time += elapsed_time;
            
        }
        result.ori_time = total_time / (test_count);
        printf("origin fprop average time: %f ms\n",result.ori_time);
        //############target############
        total_time = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        printf("%d %d %d %d size:%d\n",in_batch ,in_height, in_channels, in_width,in_elems);
        for (int i = 0; i < test_count; i++)
        {
            cudaEventRecord(start, 0);
            //优化后
            test_tmp_kernel_opt(input, output_target, weight,bias, in_batch ,in_height, in_channels, in_width,out_batch ,out_height ,out_channels , out_width,in_elems,out_elems,stream);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float elapsed_time = 0.0f;
            cudaEventElapsedTime(&elapsed_time, start, stop);
            
            total_time += elapsed_time;
        }

        result.opt_time = total_time / (test_count);
        printf("opt   fprop average time: %f ms\n", result.opt_time);

        cudaDeviceSynchronize();
 
    // 将数据从 GPU 拷贝回 CPU
    cudaMemcpy(output_cpu, output, sizeof(T) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_target_cpu, output_target, sizeof(T) * out_elems, cudaMemcpyDeviceToHost);

    // 比较两个输出是否一致
    result.time_rate = result.opt_time/result.ori_time;
    result.result_is_right = checkresult<T>(output_cpu,output_target_cpu, out_elems);


    // 释放内存
    free(input_cpu);
    free(output_cpu);
    free(output_target_cpu);
    free(weight_cpu);
    free(bias_cpu);

    cudaFree(input);
    cudaFree(output);
    cudaFree(output_target);
    cudaFree(weight);
    cudaFree(bias);

    return result;
}

int main() {
    //单输入的concat
    ResultStruct result1 = test_tmp<float>({16,256,64,256},{16,256,64,256}); // input NHCW 和output NHCW 

  
    printf("%lf ",result1.ori_time);
    printf("<time_before_opt>%f ms</time_before_opt>\n",result1.ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", result1.opt_time);
    printf("<runtime_ratio>%f</runtime_ratio>\n", result1.opt_time/result1.ori_time);
    if (result1.result_is_right)
    {
      printf("<precision>True</precision>\n");
    }
    else
    {
      printf("<precision>False</precision>\n");
    }
}