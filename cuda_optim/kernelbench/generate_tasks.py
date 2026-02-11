#!/usr/bin/env python3
"""
从 cuda_eval_code 四个等级文件夹生成 kernelbench 任务代码。
任务名带前缀: level1_, level2_, level3_, level4_
"""

import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_LIST = REPO_ROOT / "cuda_optim" / "examples" / "kernelbench_pplformat" / "run_list"
KERNELBENCH = REPO_ROOT / "cuda_optim" / "kernelbench"
COMMON_H = RUN_LIST / "19_ReLU_run" / "19_ReLU_run" / "inc" / "common.h"
TEMPLATE_MAKEFILE = RUN_LIST / "19_ReLU_run" / "19_ReLU_run" / "Makefile"
TEMPLATE_RUN_SH = RUN_LIST / "19_ReLU_run" / "19_ReLU_run" / "run.sh"

# 四个等级的配置: (cuda_eval_code 路径, 任务名前缀)
LEVEL_CONFIG = {
    1: (REPO_ROOT / "cuda_eval_code" / "claude-sonnet-4-5_hf-level-1" / "level_1", "level1_"),
    2: (REPO_ROOT / "cuda_eval_code" / "claude-sonnet-4-5_hf-level-2_2" / "level_2", "level2_"),
    3: (REPO_ROOT / "cuda_eval_code" / "claude-sonnet-4-5_hf-level-3_2" / "level_3", "level3_"),
    4: (REPO_ROOT / "cuda_eval_code" / "claude-sonnet-4-5_hf-level-4_2" / "level_4", "level4_"),
}

TASK_NAME_MAPPING = {
    "26_GELU_": "26_GELU__run",
    "27_SELU_": "27_SELU__run",
    "36_RMSNorm_": "36_RMSNorm__run",
    "38_L1Norm_": "38_L1Norm__run",
    "39_L2Norm_": "39_L2Norm__run",
}


def get_run_list_name(eval_task_name: str) -> str | None:
    name = eval_task_name.rstrip("/")
    if name in TASK_NAME_MAPPING:
        return TASK_NAME_MAPPING[name]
    run_name = name + "_run"
    if (RUN_LIST / run_name).exists():
        return run_name
    return None


def find_kernel_names(content: str) -> list[str]:
    pattern = r"__global__\s+void\s+(\w+)\s*\("
    return re.findall(pattern, content)


def convert_to_opt(content: str) -> str:
    result = content
    kernel_names = find_kernel_names(content)
    for kn in kernel_names:
        if not kn.endswith("_opt") and not kn.endswith("_ori"):
            result = re.sub(rf"\b{re.escape(kn)}\b", kn + "_opt", result)
    result = result.replace("test_tmp_kernel_ori", "test_tmp_kernel_opt")
    return result


def convert_to_check(content: str) -> str:
    result = content
    kernel_names = find_kernel_names(content)
    for kn in kernel_names:
        if not kn.endswith("_opt") and not kn.endswith("_ori"):
            result = re.sub(rf"\b{re.escape(kn)}\b", kn + "_ori", result)
    return result


def ensure_warp_size(content: str) -> str:
    if "WARP_SIZE" in content and "#define WARP_SIZE" not in content:
        if "#include" in content:
            first_include = content.find("#include")
            insert_pos = content.find("\n", first_include) + 1
            return content[:insert_pos] + "\n#ifndef WARP_SIZE\n#define WARP_SIZE 32\n#endif\n" + content[insert_pos:]
    return content


def parse_test_sig(content: str) -> dict:
    """解析 test_tmp_kernel_ori 的签名和调用方式"""
    sig = {
        "standard_12": True,
        "has_gamma_beta": False,
        "has_extra_input": False,  # input + in_elems 用于 bias 等
    }
    # GroupNorm: test_tmp_kernel_ori(..., stream, gamma, beta, num_groups, eps)
    if re.search(r"stream\s*,\s*T\s*\*\s*gamma", content) or re.search(r"stream\s*,\s*gamma\s*,\s*beta", content):
        sig["standard_12"] = False
        sig["has_gamma_beta"] = True
    if "input + in_elems" in content or "input+in_elems" in content.replace(" ", ""):
        sig["has_extra_input"] = True
    return sig


def extract_test_sizes(content: str) -> tuple[list[int] | None, list[int] | None]:
    match = re.search(
        r"test_tmp<float>\s*\(\s*\{([^}]+)\}\s*,\s*\{([^}]+)\}\s*\)",
        content,
    )
    if match:
        in_str, out_str = match.groups()
        input_size = [int(x.strip()) for x in in_str.split(",") if x.strip()]
        output_size = [int(x.strip()) for x in out_str.split(",") if x.strip()]
        return input_size, output_size
    return None, None


def write_tmp_test_standard(
    path: Path,
    input_size: list[int],
    output_size: list[int],
    extra_input_elems: int = 0,
) -> None:
    """标准 12 参数接口的 tmp_test"""
    in_str = ",".join(map(str, input_size))
    out_str = ",".join(map(str, output_size))
    if extra_input_elems:
        total_in = "in_elems + " + str(extra_input_elems)
        alloc_input = f"""
    int total_input_elems = {total_in};
    input_cpu = (T *)malloc(sizeof(T) * total_input_elems);
    cudaMalloc((void **)&input, sizeof(T) * total_input_elems);"""
        init_extra = """
    for(int i = in_elems; i < total_input_elems; i++) { input_cpu[i] = (i % 127) * 0.01f; }"""
        memcpy_size = f"({total_in})"
    else:
        alloc_input = """
    input_cpu = (T *)malloc(sizeof(T) * in_elems);
    cudaMalloc((void **)&input, sizeof(T) * in_elems);"""
        init_extra = ""
        memcpy_size = "in_elems"

    content = f'''#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "common.h"
#include "tmp_check.cuh"
#include "tmp_use.cuh"
struct ResultStruct {{
    float ori_time;
    float opt_time;
    float time_rate;
    bool result_is_right;
}};

template <typename T>
ResultStruct test_tmp(std::vector<int> input_size, std::vector<int> output_size) {{
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
    output_target_cpu = (T *)malloc(sizeof(T) * out_elems);{alloc_input}
    cudaMalloc((void **)&output, sizeof(T) * out_elems);
    cudaMalloc((void **)&output_target, sizeof(T) * out_elems);
    memset(output_cpu, 1, sizeof(T) * out_elems);
    memset(output_target_cpu, 2, sizeof(T) * out_elems);
    cudaMemset(output, 1, sizeof(T) * out_elems);
    cudaMemset(output_target, 2, sizeof(T) * out_elems);

    for(int i = 0; i < in_elems; i++) {{ input_cpu[i] = (i*7)%127; }}{init_extra}
    cudaMemcpy(input, input_cpu, sizeof(T) * {memcpy_size}, cudaMemcpyHostToDevice);

    float total_time = 0.0;
    int test_count = 1000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 5; i++)
        test_tmp_kernel_ori(input, output, in_batch, in_height, in_channels, in_width, out_batch, out_height, out_channels, out_width, in_elems, out_elems, stream);
    for (int i = 0; i < test_count; i++) {{
        cudaEventRecord(start, 0);
        test_tmp_kernel_ori(input, output, in_batch, in_height, in_channels, in_width, out_batch, out_height, out_channels, out_width, in_elems, out_elems, stream);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t;
        cudaEventElapsedTime(&t, start, stop);
        total_time += t;
    }}
    result.ori_time = total_time / test_count;
    total_time = 0;
    for (int i = 0; i < test_count; i++) {{
        cudaEventRecord(start, 0);
        test_tmp_kernel_opt(input, output_target, in_batch, in_height, in_channels, in_width, out_batch, out_height, out_channels, out_width, in_elems, out_elems, stream);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t;
        cudaEventElapsedTime(&t, start, stop);
        total_time += t;
    }}
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
}}

int main() {{
    ResultStruct result1 = test_tmp<float>({{{in_str}}}, {{{out_str}}});
    printf("%lf ", result1.ori_time);
    printf("<time_before_opt>%f ms</time_before_opt>\\n", result1.ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\\n", result1.opt_time);
    printf("<runtime_ratio>%f</runtime_ratio>\\n", result1.opt_time / result1.ori_time);
    printf("<precision>%s</precision>\\n", result1.result_is_right ? "True" : "False");
}}
'''
    path.write_text(content, encoding="utf-8")


def write_tmp_test_groupnorm(path: Path, input_size: list[int], output_size: list[int]) -> None:
    """GroupNorm 专用 tmp_test (gamma, beta, num_groups, eps)"""
    in_str = ",".join(map(str, input_size))
    out_str = ",".join(map(str, output_size))
    content = f'''#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include "common.h"
#include "tmp_check.cuh"
#include "tmp_use.cuh"
struct ResultStruct {{
    float ori_time;
    float opt_time;
    float time_rate;
    bool result_is_right;
}};

template <typename T>
ResultStruct test_tmp(std::vector<int> input_size, std::vector<int> output_size) {{
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
    int num_groups = 8;
    float eps = 1e-5f;

    T *input = nullptr;
    T *output = nullptr;
    T *output_target = nullptr;
    T *gamma = nullptr;
    T *beta = nullptr;
    T *input_cpu = nullptr;
    T *output_cpu = nullptr;
    T *output_target_cpu = nullptr;
    T *gamma_cpu = nullptr;
    T *beta_cpu = nullptr;

    input_cpu = (T *)malloc(sizeof(T) * in_elems);
    output_cpu = (T *)malloc(sizeof(T) * out_elems);
    output_target_cpu = (T *)malloc(sizeof(T) * out_elems);
    gamma_cpu = (T *)malloc(sizeof(T) * in_channels);
    beta_cpu = (T *)malloc(sizeof(T) * in_channels);

    cudaMalloc(&input, sizeof(T) * in_elems);
    cudaMalloc(&output, sizeof(T) * out_elems);
    cudaMalloc(&output_target, sizeof(T) * out_elems);
    cudaMalloc(&gamma, sizeof(T) * in_channels);
    cudaMalloc(&beta, sizeof(T) * in_channels);

    memset(output_cpu, 1, sizeof(T) * out_elems);
    memset(output_target_cpu, 2, sizeof(T) * out_elems);
    cudaMemset(output, 1, sizeof(T) * out_elems);
    cudaMemset(output_target, 2, sizeof(T) * out_elems);

    for (int i = 0; i < in_elems; i++) {{ input_cpu[i] = (i*7)%127; }}
    for (int i = 0; i < in_channels; i++) {{ gamma_cpu[i] = 1.0f; beta_cpu[i] = 0.0f; }}

    cudaMemcpy(input, input_cpu, sizeof(T) * in_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(gamma, gamma_cpu, sizeof(T) * in_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(beta, beta_cpu, sizeof(T) * in_channels, cudaMemcpyHostToDevice);

    float total_time = 0.0;
    int test_count = 1000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 5; i++)
        test_tmp_kernel_ori(input, output, in_batch, in_height, in_channels, in_width, out_batch, out_height, out_channels, out_width, in_elems, out_elems, stream, gamma, beta, num_groups, eps);
    for (int i = 0; i < test_count; i++) {{
        cudaEventRecord(start, 0);
        test_tmp_kernel_ori(input, output, in_batch, in_height, in_channels, in_width, out_batch, out_height, out_channels, out_width, in_elems, out_elems, stream, gamma, beta, num_groups, eps);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t;
        cudaEventElapsedTime(&t, start, stop);
        total_time += t;
    }}
    result.ori_time = total_time / test_count;
    total_time = 0;
    for (int i = 0; i < test_count; i++) {{
        cudaEventRecord(start, 0);
        test_tmp_kernel_opt(input, output_target, in_batch, in_height, in_channels, in_width, out_batch, out_height, out_channels, out_width, in_elems, out_elems, stream, gamma, beta, num_groups, eps);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t;
        cudaEventElapsedTime(&t, start, stop);
        total_time += t;
    }}
    result.opt_time = total_time / test_count;
    cudaDeviceSynchronize();
    cudaMemcpy(output_cpu, output, sizeof(T) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_target_cpu, output_target, sizeof(T) * out_elems, cudaMemcpyDeviceToHost);
    result.time_rate = result.opt_time / result.ori_time;
    result.result_is_right = checkresult<T>(output_cpu, output_target_cpu, out_elems);

    free(input_cpu);
    free(output_cpu);
    free(output_target_cpu);
    free(gamma_cpu);
    free(beta_cpu);
    cudaFree(input);
    cudaFree(output);
    cudaFree(output_target);
    cudaFree(gamma);
    cudaFree(beta);
    return result;
}}

int main() {{
    ResultStruct result1 = test_tmp<float>({{{in_str}}}, {{{out_str}}});
    printf("%lf ", result1.ori_time);
    printf("<time_before_opt>%f ms</time_before_opt>\\n", result1.ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\\n", result1.opt_time);
    printf("<runtime_ratio>%f</runtime_ratio>\\n", result1.opt_time / result1.ori_time);
    printf("<precision>%s</precision>\\n", result1.result_is_right ? "True" : "False");
}}
'''
    path.write_text(content, encoding="utf-8")


def write_tmp_test_2d(path: Path, input_size: list[int], output_size: list[int]) -> None:
    """2D 布局 (cumsum 等)"""
    if not input_size or len(input_size) < 2:
        input_size = [128, 4000]
        output_size = [128, 4000]
    in_b, in_c = input_size[0], input_size[1]
    out_b, out_c = output_size[0], output_size[1]
    in_str = f"{in_b},{in_c},1,1"
    out_str = f"{out_b},{out_c},1,1"
    write_tmp_test_standard(path, [in_b, in_c, 1, 1], [out_b, out_c, 1, 1])


def get_default_sizes(level: int, task_name: str, run_list_name: str | None) -> tuple[list[int], list[int]]:
    """获取默认输入输出尺寸"""
    if run_list_name and (RUN_LIST / run_list_name / run_list_name / "src" / "tmp_test.cu").exists():
        tc = (RUN_LIST / run_list_name / run_list_name / "src" / "tmp_test.cu").read_text(encoding="utf-8", errors="replace")
        sizes = extract_test_sizes(tc)
        if sizes[0] and sizes[1]:
            return sizes[0], sizes[1]
    if level == 1:
        return [1024, 16384, 1, 1], [1024, 16384, 1, 1]
    if level == 2:
        return [16, 64, 64, 64], [16, 64, 64, 64]
    if level == 3:
        return [8, 256, 256, 256], [8, 256, 256, 256]
    if level == 4:
        return [1, 1024, 1024, 1], [1, 1024, 1024, 1]
    return [16, 64, 64, 64], [16, 64, 64, 64]


def process_task(level: int, eval_task_name: str) -> bool:
    eval_path, prefix = LEVEL_CONFIG[level]
    name = eval_task_name.rstrip("/")
    ori_cu = eval_path / name / "tmp_ori.cu"

    if not ori_cu.exists():
        return False

    ori_content = ori_cu.read_text(encoding="utf-8", errors="replace")
    run_list_name = get_run_list_name(name) if level == 1 else None
    task_run_name = (prefix + (run_list_name or name + "_run")).replace("_run_run", "_run")
    out_path = KERNELBENCH / task_run_name / task_run_name
    run_path = RUN_LIST / run_list_name / run_list_name if run_list_name else None

    inc_path = out_path / "inc"
    src_path = out_path / "src"
    inc_path.mkdir(parents=True, exist_ok=True)
    src_path.mkdir(parents=True, exist_ok=True)

    # ori_content 已在上面读取
    if "std::vector" in ori_content and "#include <vector>" not in ori_content:
        ori_content = "#include <vector>\n" + ori_content
    ori_content = ensure_warp_size(ori_content)

    opt_content = convert_to_opt(ori_content)
    (inc_path / "tmp_ori.cuh").write_text(opt_content, encoding="utf-8")
    (inc_path / "tmp_use.cuh").write_text(opt_content, encoding="utf-8")
    (inc_path / "tmp_check.cuh").write_text(convert_to_check(ori_content), encoding="utf-8")

    shutil.copy2(COMMON_H, inc_path / "common.h")

    sig = parse_test_sig(ori_content)
    input_size, output_size = get_default_sizes(level, name, run_list_name)

    if sig["has_gamma_beta"]:
        write_tmp_test_groupnorm(src_path / "tmp_test.cu", input_size, output_size)
    elif sig["has_extra_input"]:
        extra = input_size[2] if len(input_size) > 2 else 64
        write_tmp_test_standard(src_path / "tmp_test.cu", input_size, output_size, extra_input_elems=extra)
    elif run_list_name and run_path and (run_path / "src" / "tmp_test.cu").exists():
        run_test = (run_path / "src" / "tmp_test.cu").read_text(encoding="utf-8", errors="replace")
        if "in_elems = in_batch * in_channels" in run_test and "in_height * in_channels" not in run_test:
            write_tmp_test_2d(src_path / "tmp_test.cu", input_size, output_size)
        elif "in_elems,out_elems,stream)" in run_test and "stream,kernel_size" not in run_test and "stream,group_num" not in run_test:
            shutil.copy2(run_path / "src" / "tmp_test.cu", src_path / "tmp_test.cu")
        else:
            write_tmp_test_standard(src_path / "tmp_test.cu", input_size, output_size)
    else:
        write_tmp_test_standard(src_path / "tmp_test.cu", input_size, output_size)

    shutil.copy2(TEMPLATE_MAKEFILE, out_path / "Makefile")
    if run_path and (run_path / "run.sh").exists():
        shutil.copy2(run_path / "run.sh", out_path / "run.sh")
    else:
        shutil.copy2(TEMPLATE_RUN_SH, out_path / "run.sh")
    if run_path and (run_path / "maca_jxie.sh").exists():
        shutil.copy2(run_path / "maca_jxie.sh", out_path / "maca_jxie.sh")

    return True


def main():
    print("从 cuda_eval_code 四个等级生成 kernelbench 任务...")
    print(f"目标: {KERNELBENCH}\n")
    KERNELBENCH.mkdir(parents=True, exist_ok=True)

    total = 0
    for level in [1, 2, 3, 4]:
        eval_path, prefix = LEVEL_CONFIG[level]
        if not eval_path.exists():
            print(f"跳过 level_{level}: 路径不存在 {eval_path}")
            continue
        print(f"[Level {level}] {eval_path}")
        for item in sorted(eval_path.iterdir()):
            if item.is_dir() and (item / "tmp_ori.cu").exists():
                ori = (item / "tmp_ori.cu").read_text(encoding="utf-8", errors="replace")
                if "test_tmp_kernel_ori" not in ori:
                    print(f"  跳过: {item.name} (tmp_ori.cu 缺少 test_tmp_kernel_ori)")
                    continue
                if process_task(level, item.name):
                    total += 1
                    print(f"  已生成: {prefix}{item.name}_run")
        print()

    print(f"完成，共生成 {total} 个任务")


if __name__ == "__main__":
    main()
