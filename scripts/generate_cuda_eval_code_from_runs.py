import argparse
import json
import os
from litellm import completion
from dotenv import load_dotenv
from string import Template
from tqdm import tqdm

load_dotenv()

RUNS_ROOT = "runs"
OUTPUT_ROOT = "cuda_eval_code"
KERNEL_BENCH_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "KernelBench",
)
LEVEL_PROBLEMS = {
    1: range(1, 101),
    2: range(1, 101),
    3: range(1, 51),
    4: range(1, 21),
    "4_expand": range(1, 21),
}

MODEL_NAME = "anthropic/claude-sonnet-4-5-20250929"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--level", type=str, default=None,
                        help="处理的 level（未指定 --failed_results 时必填）")
    parser.add_argument("--failed_results", type=str, default=None,
                        help="batch_test_results.txt 路径，仅重新生成其中的失败任务")
    args = parser.parse_args()
    if args.failed_results is None and args.level is None:
        parser.error("未指定 --failed_results 时必须提供 --level")
    return args


def _level_to_subdir(level: str) -> str:
    """将 level 参数映射为 KernelBench 子目录名。"""
    if level == "4_expand":
        return "level4_expand"
    return f"level{int(level)}"


def load_eval_results(run_name: str) -> dict:
    """加载 run 目录下的 eval_results.json，不存在或解析失败则返回空 dict。"""
    path = os.path.join(RUNS_ROOT, run_name, "eval_results.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def get_eval_entry_for_sample(
    eval_results: dict, problem_id: int, sample_id: int = 0
) -> dict | None:
    """
    从 eval_results 中取出指定 problem_id、sample_id 的条目。
    支持 list 格式 [{"sample_id": 0, "compiled": ..., "correctness": ...}, ...]
    与旧版 dict 格式 {"sample_id": 0, "compiled": ..., "correctness": ...}。
    不存在则返回 None。
    """
    key = str(problem_id)
    if key not in eval_results:
        return None
    entry = eval_results[key]
    if isinstance(entry, list):
        for r in entry:
            if r.get("sample_id") == sample_id:
                return r
        return None
    if entry.get("sample_id") == sample_id:
        return entry
    return None


def get_problem_name(level: str, problem_id: int) -> str | None:
    """
    从 KernelBench 对应 level 目录下根据 problem_id 查找问题文件名（不含扩展名），
    作为目录名使用，例如 1_Square_matrix_multiplication_。
    """
    subdir = _level_to_subdir(level)
    problem_dir = os.path.join(KERNEL_BENCH_PATH, subdir)
    if not os.path.isdir(problem_dir):
        return None
    prefix = f"{problem_id}_"
    for name in os.listdir(problem_dir):
        if name.endswith(".py") and name.startswith(prefix):
            return os.path.splitext(name)[0]
    return None


def extract_kernel_code(
    run_name: str, level: str, problem_id: int, sample_id: int = 0
) -> str:
    kernel_path = os.path.join(
        RUNS_ROOT,
        run_name,
        f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py",
    )

    if not os.path.exists(kernel_path):
        print(f"Warning: Kernel file not found at {kernel_path}")
        return None, None, None

    code = ""
    with open(kernel_path, "r") as f:
        code = f.read()

    ############ #define macro code     ############
    macro_code = ""
    for line in code.split("\n"):
        if line.startswith("#define"):
            macro_code += line + "\n"

    ############ __global__ kernel code ############

    kernel_code = ""
    in_kernel = False
    for line in code.split("\n"):
        if line.startswith("__global__"):
            in_kernel = True
            kernel_code += line + "\n"
        if in_kernel:
            kernel_code += line + "\n"
        if line.startswith("}"):
            kernel_code += line + "\n"
            in_kernel = False

    ############ torch::Tensor torch entry code ####

    entry_code = ""
    in_entry = False
    for line in code.split("\n"):
        if line.startswith("torch::Tensor"):
            in_entry = True
            entry_code += line + "\n"
        if in_entry:
            entry_code += line + "\n"
        if line.startswith("}"):
            entry_code += line + "\n"
            in_entry = False

    return macro_code, kernel_code, entry_code


# one-shot template for generate c++ entry code
one_shot_template = Template(
    """You are a CUDA expert. Your task is to generate a c++ entry code for the given kernel code and torch::Tensor entry code. You should follow the following rules:
- You should use the original kernel code and torch::Tensor entry code as a reference.
- You should generate the entry code that is compatible with the torch::Tensor entry code.
- You should write the entry code with the original kernel code
- Do not change the original kernel code.
- Do not change the original macro code.
- Do not change the original #include code.
- Do not change the original #define code.
- You should only output the entry code, no other text.


You are given the following kernel code and torch::Tensor entry code:
```cpp
$macro_code
$kernel_code
$entry_code
```

Your example entry code:
```cpp
__global__ void leaky_relu_kernel_ori(const float* x, float* y, float negative_slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = (x[idx] > 0.0) ? x[idx] : x[idx] * negative_slope;
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
    int  size = in_elems;
    float negative_slope = 0.01;
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    leaky_relu_kernel_ori<<<num_blocks, block_size>>>(input, output,negative_slope, size);
}
```
"""
)


def make_prompt(macro_code: str, kernel_code: str, entry_code: str) -> str:
    one_shot_prompt = one_shot_template.substitute(
        macro_code=macro_code, kernel_code=kernel_code, entry_code=entry_code
    )
    return one_shot_prompt


def parse_failed_tasks(results_path: str) -> dict[str, list[int]]:
    """
    解析 batch_test_results.txt，提取失败任务的 (level, problem_id)。

    返回: {level_str: [problem_id, ...]}，例如 {"1": [23, 26], "2": [10, 16]}
    """
    import re
    failed: dict[str, list[int]] = {}
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "COMPILE_ERROR" in line or "RUNTIME_ERROR" in line or "TIMEOUT" in line:
                m = re.match(r"^level(\d+)_(\d+)_", line)
                if m:
                    level_str = m.group(1)
                    problem_id = int(m.group(2))
                    failed.setdefault(level_str, [])
                    if problem_id not in failed[level_str]:
                        failed[level_str].append(problem_id)
    for level_str in failed:
        failed[level_str].sort()
    return failed


def generate_for_problems(
    run_name: str,
    level: str,
    problem_ids: list[int],
    sample_id: int,
    skip_eval_check: bool = False,
) -> dict:
    """
    对指定的 (level, problem_ids) 列表执行代码生成。

    skip_eval_check=True 时跳过 eval_results 过滤（用于重新生成失败任务）。
    返回 summary dict。
    """
    eval_results = {}
    if not skip_eval_check:
        eval_results = load_eval_results(run_name)
        if not eval_results:
            print(
                f"Warning: 未找到或无法解析 runs/{run_name}/eval_results.json，"
                "将不进行任何转换。"
            )

    output_dir = os.path.join(OUTPUT_ROOT, run_name, f"level_{level}")
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "success": [],
        "compile_failed": [],
        "output_error": [],
        "no_eval_result": [],
        "kernel_missing": [],
        "conversion_failed": [],
    }

    for problem_id in tqdm(
        problem_ids,
        total=len(problem_ids),
        desc=f"生成 Level {level} CUDA 评估代码",
        unit="题",
        ncols=80,
    ):
        problem_name = get_problem_name(level, problem_id)
        display_name = problem_name if problem_name else f"problem_{problem_id}"

        if not skip_eval_check:
            entry = get_eval_entry_for_sample(eval_results, problem_id, sample_id)
            if entry is None:
                summary["no_eval_result"].append(display_name)
                continue
            if not entry.get("compiled", False):
                summary["compile_failed"].append(display_name)
                continue
            if not entry.get("correctness", False):
                summary["output_error"].append(display_name)
                continue

        macro_code, kernel_code, entry_code = extract_kernel_code(
            run_name, level, problem_id, sample_id
        )
        if kernel_code is None or entry_code is None:
            summary["kernel_missing"].append(display_name)
            continue

        prompt = make_prompt(macro_code, kernel_code, entry_code)
        try:
            response = completion(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32768,
                temperature=0.0,
                api_key=ANTHROPIC_API_KEY,
            )
            eval_code = response.choices[0].message.content
        except Exception as e:
            print(f"Warning: 调用模型失败 problem_id={problem_id}, error={e}")
            summary["conversion_failed"].append(display_name)
            continue

        if not eval_code.startswith("```cpp"):
            print(
                f"Warning: Generated eval code does not valid c++ code, "
                f"skipping problem {problem_id}"
            )
            summary["conversion_failed"].append(display_name)
            continue
        eval_code = eval_code.replace("```cpp", "").replace("```", "").strip()

        dir_name = problem_name if problem_name else f"problem_{problem_id}"
        output_dir_problem = os.path.join(output_dir, dir_name)
        os.makedirs(output_dir_problem, exist_ok=True)
        with open(
            os.path.join(output_dir_problem, "tmp_ori.cu"), "w", encoding="utf-8"
        ) as f:
            f.write(eval_code)
        summary["success"].append(display_name)

    return summary


def main():
    args = parse_args()
    run_name = args.run_name
    sample_id = 0

    if args.failed_results:
        # ---------- 失败任务重新生成模式 ----------
        failed_tasks = parse_failed_tasks(args.failed_results)
        if not failed_tasks:
            print(f"未从 {args.failed_results} 中发现失败任务。")
            return

        print(f"从 {args.failed_results} 解析到失败任务:")
        for lvl in sorted(failed_tasks):
            pids = failed_tasks[lvl]
            print(f"  Level {lvl}: {len(pids)} 个 — {pids}")

        # 若同时指定了 --level，则只处理该 level 的失败任务
        if args.level:
            lvl = args.level
            if lvl not in failed_tasks:
                print(f"Level {lvl} 没有失败任务。")
                return
            levels_to_process = {lvl: failed_tasks[lvl]}
        else:
            levels_to_process = failed_tasks

        all_summaries = {}
        for level_str in sorted(levels_to_process):
            problem_ids = levels_to_process[level_str]
            print(f"\n===== 重新生成 Level {level_str} ({len(problem_ids)} 个任务) =====")
            summary = generate_for_problems(
                run_name, level_str, problem_ids, sample_id, skip_eval_check=True
            )
            all_summaries[f"level_{level_str}"] = summary

            output_dir = os.path.join(OUTPUT_ROOT, run_name, f"level_{level_str}")
            summary_path = os.path.join(output_dir, "regeneration_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"  重新生成总结已写入: {summary_path}")
            print(f"  成功: {len(summary['success'])}, "
                  f"kernel 缺失: {len(summary['kernel_missing'])}, "
                  f"转换失败: {len(summary['conversion_failed'])}")

    else:
        # ---------- 原始全量生成模式 ----------
        level = args.level
        level_key = level if level == "4_expand" else int(level)
        if level_key not in LEVEL_PROBLEMS:
            raise ValueError(f"不支持的 level: {level}，可选: 1, 2, 3, 4, 4_expand")

        problem_ids = list(LEVEL_PROBLEMS[level_key])
        summary = generate_for_problems(
            run_name, level, problem_ids, sample_id, skip_eval_check=False
        )

        output_dir = os.path.join(OUTPUT_ROOT, run_name, f"level_{level}")
        summary_path = os.path.join(output_dir, "conversion_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"转换总结已写入: {summary_path}")


if __name__ == "__main__":
    main()
