#!/usr/bin/env python3
import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import warnings
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

# Suppress known benign pydantic serializer warnings emitted by some LiteLLM
# providers when response schemas differ between streaming/non-streaming shapes.
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:.*",
    category=UserWarning,
)

try:
    from litellm import completion
except ImportError:
    completion = None

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "runs"
CUDA_EVAL_ROOT = REPO_ROOT / "cuda_eval_code"
KERNELBENCH_TASK_ROOT = REPO_ROOT / "cuda_optim" / "kernelbench"
KERNELBENCH_META_ROOT = REPO_ROOT / "KernelBench"

LEVEL_PROBLEMS = {
    "1": list(range(1, 101)),
    "2": list(range(1, 101)),
    "3": list(range(1, 51)),
    "4": list(range(1, 21)),
    "level4_expand": list(range(1, 21)),
}

TASK_NAME_MAPPING = {
    "26_GELU_": "26_GELU__run",
    "27_SELU_": "27_SELU__run",
    "36_RMSNorm_": "36_RMSNorm__run",
    "38_L1Norm_": "38_L1Norm__run",
    "39_L2Norm_": "39_L2Norm__run",
}

ONE_SHOT_TEMPLATE = Template(
    """You are a CUDA expert. Your task is to generate CUDA C++ entry code for kernelbench evaluation.
Follow these rules:
- Keep original kernel logic.
- Keep original macro code.
- Keep includes and defines as needed, but ONLY CUDA/C++ standard headers.
- DO NOT include torch headers (e.g. torch/extension.h, ATen, pybind11, Python.h).
- DO NOT output torch::Tensor wrappers or PYBIND11_MODULE code.
- Output only valid C++ code wrapped by ```cpp ... ```.
- Provide a callable function:
template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream
)

Example output code:
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

Given code:
```cpp
$macro_code
$kernel_code
$entry_code
```
"""
)


@dataclass
class TaskState:
    level: str
    problem_id: int
    problem_name: str
    task_dir_name: str
    status: str = "pending"  # pending | success | failed
    attempt: int = 0
    generated_code: str = ""
    error_history: list[dict[str, Any]] = field(default_factory=list)
    precision: str = ""
    runtime_ratio: str = ""
    time_before: str = ""
    time_after: str = ""
    compile_log_path: str = ""
    run_log_path: str = ""


class GpuPool:
    def __init__(self, num_gpus: int):
        self._q: queue.Queue[int] = queue.Queue()
        for i in range(num_gpus):
            self._q.put(i)

    @contextmanager
    def acquire(self):
        gpu_id = self._q.get()
        try:
            yield gpu_id
        finally:
            self._q.put(gpu_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--level", type=str, default="1,2,3,4,level4_expand")
    parser.add_argument("--failed_results", type=str, default=None)
    parser.add_argument("--skip_eval_check", action="store_true")
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--parallel_gen", type=int, default=4)
    parser.add_argument(
        "--compile_workers", type=int, default=max(2, (os.cpu_count() or 4) // 2)
    )
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--nvcc_arch", type=str, default="sm_89")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_tokens", type=int, default=131072)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--api_timeout",
        type=int,
        default=120,
        help="LLM API request timeout in seconds for each completion call.",
    )
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--common_h",
        type=str,
        default="",
        help="Path to common.h template. If empty, auto-detect from kernelbench tasks.",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default=str(REPO_ROOT / "scripts" / "pipeline_reports"),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Resume from an existing report directory. "
            "Example: scripts/pipeline_reports/<run_name>_<timestamp>"
        ),
    )
    parser.add_argument(
        "--kernelbench_namespace",
        type=str,
        default="",
        help=(
            "Output namespace under cuda_optim/kernelbench/. "
            "If empty, auto-derived from run_name by taking prefix before first underscore, "
            "or full run_name when no underscore."
        ),
    )
    parser.add_argument(
        "--llm_trace_subdir",
        type=str,
        default="llm_traces",
        help="Subdirectory name under report dir for per-task LLM dialogue traces.",
    )
    parser.add_argument(
        "--disable_llm_trace",
        action="store_true",
        help="Disable writing per-task LLM prompt/response trace files.",
    )
    return parser.parse_args()


def detect_num_gpus() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r.returncode != 0:
            return 1
        lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        return max(1, len(lines))
    except Exception:
        return 1


def parse_levels(level_arg: str) -> list[str]:
    levels = [x.strip() for x in level_arg.split(",") if x.strip()]
    for lvl in levels:
        if lvl not in LEVEL_PROBLEMS:
            raise ValueError(
                f"Unsupported level: {lvl}, expected one of {list(LEVEL_PROBLEMS.keys())}"
            )
    return levels


def derive_kernelbench_namespace(run_name: str, override: str) -> str:
    if override and override.strip():
        return override.strip().strip("/")
    if "_" in run_name:
        return run_name.split("_", 1)[0].strip()
    return run_name.strip()


def find_common_h(path_arg: str) -> Path:
    if path_arg:
        p = Path(path_arg).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--common_h not found: {p}")
        return p

    candidates = [
        KERNELBENCH_TASK_ROOT
        / "level1_19_ReLU_run"
        / "level1_19_ReLU_run"
        / "inc"
        / "common.h",
        KERNELBENCH_TASK_ROOT
        / "level1_1_Square_matrix_multiplication__run"
        / "level1_1_Square_matrix_multiplication__run"
        / "inc"
        / "common.h",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Cannot auto-detect common.h. Please pass --common_h.")


def load_eval_results(run_name: str) -> dict:
    p = RUNS_ROOT / run_name / "eval_results.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_eval_entry_for_sample(
    eval_results: dict, problem_id: int, sample_id: int = 0
) -> dict | None:
    key = str(problem_id)
    if key not in eval_results:
        return None
    entry = eval_results[key]
    if isinstance(entry, list):
        for item in entry:
            if item.get("sample_id") == sample_id:
                return item
        return None
    if isinstance(entry, dict) and entry.get("sample_id") == sample_id:
        return entry
    return None


def parse_failed_tasks(results_path: Path) -> dict[str, list[int]]:
    failed: dict[str, list[int]] = {}
    for line in results_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if "COMPILE_ERROR" in s or "RUNTIME_ERROR" in s or "TIMEOUT" in s:
            m = re.match(r"^level(\d+)_(\d+)_", s)
            if not m:
                continue
            lvl = m.group(1)
            pid = int(m.group(2))
            failed.setdefault(lvl, [])
            if pid not in failed[lvl]:
                failed[lvl].append(pid)
    for lvl in failed:
        failed[lvl].sort()
    return failed


def get_problem_name(level: str, problem_id: int) -> str:
    level_dir = KERNELBENCH_META_ROOT / f"level{level}"
    if level_dir.exists():
        prefix = f"{problem_id}_"
        for name in os.listdir(level_dir):
            if name.endswith(".py") and name.startswith(prefix):
                return Path(name).stem
    return f"{problem_id}_unknown"


def make_task_dir_name(level: str, problem_name: str) -> str:
    if level == "1" and problem_name in TASK_NAME_MAPPING:
        return f"level1_{TASK_NAME_MAPPING[problem_name]}"
    return f"level{level}_{problem_name}_run".replace("_run_run", "_run")


def extract_code_blocks_by_signature(source: str, signature: str) -> str:
    lines = source.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith(signature):
            brace = 0
            seen = False
            while i < len(lines):
                line = lines[i]
                out.append(line)
                brace += line.count("{")
                brace -= line.count("}")
                if "{" in line:
                    seen = True
                i += 1
                if seen and brace <= 0:
                    break
            out.append("")
        else:
            i += 1
    return "\n".join(out).strip()


def extract_kernel_code(run_name: str, level: str, problem_id: int, sample_id: int = 0):
    kernel_path = (
        RUNS_ROOT
        / run_name
        / f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
    )
    if not kernel_path.exists():
        return None, None, None
    code = kernel_path.read_text(encoding="utf-8", errors="replace")
    macro_lines = [
        line for line in code.splitlines() if line.strip().startswith("#define")
    ]
    macro_code = "\n".join(macro_lines).strip()
    kernel_code = extract_code_blocks_by_signature(code, "__global__")
    entry_code = extract_code_blocks_by_signature(code, "torch::Tensor")
    if not kernel_code or not entry_code:
        return None, None, None
    return macro_code, kernel_code, entry_code


def make_prompt(
    macro_code: str, kernel_code: str, entry_code: str, task: TaskState
) -> str:
    prompt = ONE_SHOT_TEMPLATE.substitute(
        macro_code=macro_code or "",
        kernel_code=kernel_code,
        entry_code=entry_code,
    )
    if task.error_history:
        last = task.error_history[-1]
        prompt += (
            "\n\nPrevious attempt failed. Please fix errors and regenerate full code.\n"
            f"Error type: {last.get('type', 'unknown')}\n"
            f"Error message:\n{last.get('message', '')}\n\n"
            "Constraints:\n"
            "- Use CUDA keywords __global__ / __device__ / __host__ directly.\n"
            "- Avoid __attribute__((global)) or __attribute__((device)).\n"
            "- Ensure braces and templates are closed.\n"
            "- Declare all variables before use.\n"
            "- Include needed headers for used symbols (e.g. cfloat for FLT_MAX).\n"
        )
        if task.generated_code:
            prompt += f"\nLast generated code:\n```cpp\n{task.generated_code}\n```\n"
    return prompt


def extract_cpp_from_response(text: str) -> str:
    text = text.strip()
    m = re.search(r"```cpp\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```\s*(.*?)```", text, re.S)
    if m2:
        return m2.group(1).strip()
    return text


def _safe_primitive(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_primitive(x) for x in obj]
    return str(obj)


def extract_response_summary(response: Any) -> dict[str, Any]:
    """
    Extract stable, minimal response fields without calling pydantic serializers.
    This avoids PydanticSerializationUnexpectedValue warnings from model_dump().
    """
    if response is None:
        return {}

    summary: dict[str, Any] = {
        "id": _safe_primitive(getattr(response, "id", None)),
        "model": _safe_primitive(getattr(response, "model", None)),
        "created": _safe_primitive(getattr(response, "created", None)),
        "usage": _safe_primitive(getattr(response, "usage", None)),
        "choices": [],
    }

    choices = getattr(response, "choices", None) or []
    for c in choices:
        msg = getattr(c, "message", None)
        choice_item = {
            "index": _safe_primitive(getattr(c, "index", None)),
            "finish_reason": _safe_primitive(getattr(c, "finish_reason", None)),
            "message": {
                "role": _safe_primitive(getattr(msg, "role", None)),
                "content": _safe_primitive(getattr(msg, "content", None)),
                "reasoning_content": _safe_primitive(
                    getattr(msg, "reasoning_content", None)
                    or getattr(msg, "reasoning", None)
                    or getattr(msg, "thinking", None)
                ),
                "provider_specific_fields": _safe_primitive(
                    getattr(msg, "provider_specific_fields", None)
                ),
            },
        }
        summary["choices"].append(choice_item)
    return summary


def extract_reasoning_content(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""

        # Common fields on some providers.
        for key in ("reasoning_content", "reasoning", "thinking", "analysis"):
            v = getattr(message, key, None)
            if isinstance(v, str) and v.strip():
                return v

        # Provider specific fields, if present.
        psf = getattr(message, "provider_specific_fields", None)
        if isinstance(psf, dict):
            for key in ("reasoning_content", "reasoning", "thinking", "analysis"):
                v = psf.get(key)
                if isinstance(v, str) and v.strip():
                    return v
    except Exception:
        return ""
    return ""


def write_llm_trace(
    llm_trace_dir: Path | None,
    task: TaskState,
    prompt: str,
    model: str,
    response_obj: Any = None,
    raw_content: str = "",
    reasoning_content: str = "",
    error_message: str = "",
):
    if llm_trace_dir is None:
        return
    llm_trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = llm_trace_dir / f"{task.task_dir_name}__iter{task.attempt}.json"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "task": {
            "level": task.level,
            "problem_id": task.problem_id,
            "problem_name": task.problem_name,
            "task_dir_name": task.task_dir_name,
            "attempt": task.attempt,
        },
        "model": model,
        "prompt": prompt,
        "raw_content": raw_content,
        "reasoning_content": reasoning_content,
        "error": error_message,
        "response": extract_response_summary(response_obj),
    }
    trace_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _remove_function_block_by_prefix(code: str, prefix_regex: str) -> str:
    lines = code.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(prefix_regex, line.strip()):
            brace = line.count("{") - line.count("}")
            i += 1
            while i < len(lines):
                brace += lines[i].count("{") - lines[i].count("}")
                i += 1
                if brace <= 0:
                    break
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def sanitize_generated_code(code: str) -> str:
    # Strip markdown fences if present again.
    code = extract_cpp_from_response(code)

    # Remove torch/pybind/python specific include lines.
    forbidden_include_patterns = [
        r"^\s*#\s*include\s*<torch/.*>\s*$",
        r"^\s*#\s*include\s*<ATen/.*>\s*$",
        r"^\s*#\s*include\s*<pybind11/.*>\s*$",
        r"^\s*#\s*include\s*<Python\.h>\s*$",
    ]
    cleaned_lines: list[str] = []
    for line in code.splitlines():
        if any(re.match(pat, line) for pat in forbidden_include_patterns):
            continue
        cleaned_lines.append(line)
    code = "\n".join(cleaned_lines)

    # Remove torch::Tensor and pybind module wrappers if model still emitted them.
    code = _remove_function_block_by_prefix(code, r"^torch::Tensor\s+\w+\s*\(")
    code = _remove_function_block_by_prefix(code, r"^PYBIND11_MODULE\s*\(")

    # Normalize non-standard attributes to CUDA keywords if model emits them.
    code = code.replace("__attribute__((global))", "__global__")
    code = code.replace("__attribute__((device))", "__device__")

    return code.strip()


def find_kernel_names(content: str) -> list[str]:
    return re.findall(r"__global__\s+void\s+(\w+)\s*\(", content)


def convert_to_opt(content: str) -> str:
    result = content
    for kn in find_kernel_names(content):
        if not kn.endswith("_opt") and not kn.endswith("_ori"):
            result = re.sub(rf"\b{re.escape(kn)}\b", kn + "_opt", result)
    result = result.replace("test_tmp_kernel_ori", "test_tmp_kernel_opt")
    return result


def convert_to_check(content: str) -> str:
    result = content
    for kn in find_kernel_names(content):
        if not kn.endswith("_opt") and not kn.endswith("_ori"):
            result = re.sub(rf"\b{re.escape(kn)}\b", kn + "_ori", result)
    return result


def ensure_warp_size(content: str) -> str:
    if "WARP_SIZE" in content and "#define WARP_SIZE" not in content:
        idx = content.find("#include")
        if idx != -1:
            nl = content.find("\n", idx)
            insert = "\n#ifndef WARP_SIZE\n#define WARP_SIZE 32\n#endif\n"
            return content[: nl + 1] + insert + content[nl + 1 :]
    return content


def parse_test_sig(content: str) -> dict:
    sig = {"has_gamma_beta": False, "has_extra_input": False}
    if re.search(r"stream\s*,\s*T\s*\*\s*gamma", content) or re.search(
        r"stream\s*,\s*gamma\s*,\s*beta", content
    ):
        sig["has_gamma_beta"] = True
    if "input + in_elems" in content or "input+in_elems" in content.replace(" ", ""):
        sig["has_extra_input"] = True
    return sig


def get_default_sizes(level: str) -> tuple[list[int], list[int]]:
    if level == "1":
        return [1024, 16384, 1, 1], [1024, 16384, 1, 1]
    if level == "2":
        return [16, 64, 64, 64], [16, 64, 64, 64]
    if level == "3":
        return [8, 256, 256, 256], [8, 256, 256, 256]
    if level == "4":
        return [1, 1024, 1024, 1], [1, 1024, 1024, 1]
    return [16, 64, 64, 64], [16, 64, 64, 64]


def write_tmp_test_standard(
    path: Path,
    input_size: list[int],
    output_size: list[int],
    extra_input_elems: int = 0,
):
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

    content = f"""#include <cuda_fp16.h>
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
"""
    path.write_text(content, encoding="utf-8")


def write_tmp_test_groupnorm(path: Path, input_size: list[int], output_size: list[int]):
    in_str = ",".join(map(str, input_size))
    out_str = ",".join(map(str, output_size))
    content = f"""#include <cuda_fp16.h>
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
"""
    path.write_text(content, encoding="utf-8")


def parse_xml_tag(output: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", output, re.S)
    return m.group(1).strip() if m else ""


def collect_tasks(args) -> list[TaskState]:
    levels = parse_levels(args.level)
    selected: dict[str, list[int]] = {lvl: [] for lvl in levels}

    if args.failed_results:
        failed = parse_failed_tasks(Path(args.failed_results))
        for lvl in levels:
            selected[lvl] = [x for x in failed.get(lvl, [])]
    else:
        eval_results = load_eval_results(args.run_name)
        for lvl in levels:
            for pid in LEVEL_PROBLEMS[lvl]:
                if not args.skip_eval_check:
                    entry = get_eval_entry_for_sample(eval_results, pid, 0)
                    if (
                        entry is None
                        or (not entry.get("compiled", False))
                        or (not entry.get("correctness", False))
                    ):
                        continue
                selected[lvl].append(pid)

    tasks: list[TaskState] = []
    for lvl in levels:
        for pid in selected[lvl]:
            problem_name = get_problem_name(lvl, pid)
            tasks.append(
                TaskState(
                    level=lvl,
                    problem_id=pid,
                    problem_name=problem_name,
                    task_dir_name=make_task_dir_name(lvl, problem_name),
                )
            )
    return tasks


def generate_for_task(args, task: TaskState, llm_trace_dir: Path | None):
    if completion is None:
        task.error_history.append(
            {
                "attempt": task.attempt,
                "type": "generation",
                "message": "litellm is not installed. Install with: pip install litellm",
            }
        )
        return False

    macro_code, kernel_code, entry_code = extract_kernel_code(
        args.run_name, task.level, task.problem_id, 0
    )
    if kernel_code is None or entry_code is None:
        task.error_history.append(
            {
                "attempt": task.attempt,
                "type": "kernel_missing",
                "message": f"Kernel source not found in runs/{args.run_name} for level={task.level} problem={task.problem_id}",
            }
        )
        return False

    prompt = make_prompt(macro_code, kernel_code, entry_code, task)
    try:
        kwargs = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "timeout": args.api_timeout,
        }
        if args.api_key:
            kwargs["api_key"] = args.api_key
        response = completion(**kwargs)
        raw = response.choices[0].message.content
        reasoning = extract_reasoning_content(response)
        write_llm_trace(
            llm_trace_dir=llm_trace_dir,
            task=task,
            prompt=prompt,
            model=args.model,
            response_obj=response,
            raw_content=raw or "",
            reasoning_content=reasoning,
            error_message="",
        )
        task.generated_code = sanitize_generated_code(extract_cpp_from_response(raw))
        if "test_tmp_kernel_ori(" not in task.generated_code:
            task.error_history.append(
                {
                    "attempt": task.attempt,
                    "type": "generation_invalid",
                    "message": "Generated code missing required function test_tmp_kernel_ori(...)",
                }
            )
            return False
        if (
            "torch/extension.h" in task.generated_code
            or "PYBIND11_MODULE" in task.generated_code
        ):
            task.error_history.append(
                {
                    "attempt": task.attempt,
                    "type": "generation_invalid",
                    "message": "Generated code still contains torch/pybind specific content after sanitization.",
                }
            )
            return False
        return bool(task.generated_code.strip())
    except Exception as e:
        err_text = str(e)
        err_type = "generation"
        lowered = err_text.lower()
        if (
            "timeout" in lowered
            or "timed out" in lowered
            or "requesttimeout" in lowered
        ):
            err_type = "generation_timeout"
        write_llm_trace(
            llm_trace_dir=llm_trace_dir,
            task=task,
            prompt=prompt,
            model=args.model,
            response_obj=None,
            raw_content="",
            reasoning_content="",
            error_message=err_text,
        )
        task.error_history.append(
            {
                "attempt": task.attempt,
                "type": err_type,
                "message": err_text,
            }
        )
        return False


def write_cuda_eval_code(args, task: TaskState):
    out_dir = CUDA_EVAL_ROOT / args.run_name / f"level_{task.level}" / task.problem_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tmp_ori.cu").write_text(task.generated_code, encoding="utf-8")


def convert_to_task_dir(args, task: TaskState, common_h: Path, kernelbench_base: Path):
    code = task.generated_code
    if "std::vector" in code and "#include <vector>" not in code:
        code = "#include <vector>\n" + code
    code = ensure_warp_size(code)

    out_base = kernelbench_base / task.task_dir_name / task.task_dir_name
    inc_path = out_base / "inc"
    src_path = out_base / "src"
    inc_path.mkdir(parents=True, exist_ok=True)
    src_path.mkdir(parents=True, exist_ok=True)

    tmp_ori_code = convert_to_opt(code)
    tmp_use_code = convert_to_opt(code)
    tmp_check_code = convert_to_check(code)

    prompt = Template(
        """You are a CUDA code expert with deep expertise in C++ project development and naming conflict resolution. 

Please analyze the following code snippets to identify **identically named functions, variables, and other identifiers** (e.g., macros, structs, enums) that may cause naming collisions. To prevent naming conflicts in a C++ project, extract all code fragments that contain duplicate definitions or naming conflicts, with the following critical restriction:

### Critical Extraction Restriction:
- Only extract **functional methods** (e.g., methods marked with `__device__` such as `float xxx()`)
- **Exclude** all kernel functions marked with `__global__`, as well as the entry functions `test_tmp_kernel_opt` and `test_tmp_kernel_opi` (do not extract these at all).

### Output Requirements:
- Write the extracted conflicting code snippets inside the block: ```cpp {extracted_code} ```
- If no naming conflicts are found (after applying the above restriction), reply with the exact word: null

### Code Snippets to Check for Naming Conflicts:
```cpp
$tmp_ori_code
$tmp_use_code
$tmp_check_code
    """
    ).substitute(
        tmp_ori_code=tmp_ori_code,
        tmp_use_code=tmp_use_code,
        tmp_check_code=tmp_check_code,
    )
    kwargs = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout": args.api_timeout,
    }

    response = completion(**kwargs)
    raw = response.choices[0].message.content
    if raw == "null":
        (inc_path / "tmp_ori.cuh").write_text(tmp_ori_code, encoding="utf-8")
        (inc_path / "tmp_use.cuh").write_text(tmp_use_code, encoding="utf-8")
        (inc_path / "tmp_check.cuh").write_text(tmp_check_code, encoding="utf-8")
        shutil.copy2(common_h, inc_path / "common.h")
    else:
        extracted_code = extract_cpp_from_response(raw)
        tmp_ori_code = tmp_ori_code.replace(extracted_code, "\n")
        tmp_use_code = tmp_use_code.replace(extracted_code, "\n")
        tmp_check_code = tmp_check_code.replace(extracted_code, "\n")
        common_h_code = common_h.read_text(encoding="utf-8")
        common_h_code += "\n" + extracted_code
        (inc_path / "tmp_ori.cuh").write_text(tmp_ori_code, encoding="utf-8")
        (inc_path / "tmp_use.cuh").write_text(tmp_use_code, encoding="utf-8")
        (inc_path / "tmp_check.cuh").write_text(tmp_check_code, encoding="utf-8")
        (inc_path / "common.h").write_text(common_h_code, encoding="utf-8")

    sig = parse_test_sig(code)
    input_size, output_size = get_default_sizes(task.level)
    if sig["has_gamma_beta"]:
        write_tmp_test_groupnorm(src_path / "tmp_test.cu", input_size, output_size)
    elif sig["has_extra_input"]:
        extra = input_size[2] if len(input_size) > 2 else 64
        write_tmp_test_standard(
            src_path / "tmp_test.cu", input_size, output_size, extra_input_elems=extra
        )
    else:
        write_tmp_test_standard(src_path / "tmp_test.cu", input_size, output_size)
    return out_base


def compile_and_run(
    args, task: TaskState, task_base: Path, gpu_pool: GpuPool, logs_dir: Path
):
    src = task_base / "src" / "tmp_test.cu"
    inc = task_base / "inc"
    binary = task_base / "test_cuda"

    compile_log = logs_dir / f"{task.task_dir_name}.compile.log"
    run_log = logs_dir / f"{task.task_dir_name}.run.log"
    task.compile_log_path = str(compile_log)
    task.run_log_path = str(run_log)

    compile_cmd = [
        "nvcc",
        f"-arch={args.nvcc_arch}",
        "-I",
        str(inc),
        str(src),
        "-o",
        str(binary),
    ]
    c = subprocess.run(
        compile_cmd, capture_output=True, text=True, timeout=180, check=False
    )
    compile_log.write_text((c.stdout or "") + "\n" + (c.stderr or ""), encoding="utf-8")
    if c.returncode != 0:
        task.error_history.append(
            {
                "attempt": task.attempt,
                "type": "compile",
                "message": (c.stderr or c.stdout or "nvcc compile failed")[:4000],
            }
        )
        return False

    try:
        with gpu_pool.acquire() as gpu_id:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            r = subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env=env,
                check=False,
            )
    except subprocess.TimeoutExpired:
        task.error_history.append(
            {
                "attempt": task.attempt,
                "type": "runtime_timeout",
                "message": f"runtime timeout after {args.timeout}s",
            }
        )
        return False

    run_log.write_text((r.stdout or "") + "\n" + (r.stderr or ""), encoding="utf-8")
    if r.returncode != 0:
        task.error_history.append(
            {
                "attempt": task.attempt,
                "type": "runtime",
                "message": (r.stderr or r.stdout or f"runtime exit={r.returncode}")[
                    :4000
                ],
            }
        )
        return False

    task.precision = parse_xml_tag(r.stdout, "precision")
    task.runtime_ratio = parse_xml_tag(r.stdout, "runtime_ratio")
    task.time_before = parse_xml_tag(r.stdout, "time_before_opt")
    task.time_after = parse_xml_tag(r.stdout, "time_after_opt")
    task.status = "success"
    try:
        binary.unlink(missing_ok=True)
    except Exception:
        pass
    return True


def iteration_summary(iter_idx: int, tasks: list[TaskState]):
    pending = sum(1 for t in tasks if t.status == "pending")
    success = sum(1 for t in tasks if t.status == "success")
    failed = sum(1 for t in tasks if t.status == "failed")
    print(f"\n[iter={iter_idx}] success={success} failed={failed} pending={pending}")


def write_report(tasks: list[TaskState], report_base: Path):
    report_base.mkdir(parents=True, exist_ok=True)
    json_path = report_base / "pipeline_report.json"
    txt_path = report_base / "pipeline_report.txt"

    payload = []
    for t in tasks:
        payload.append(
            {
                "level": t.level,
                "problem_id": t.problem_id,
                "problem_name": t.problem_name,
                "task_dir_name": t.task_dir_name,
                "status": t.status,
                "attempt": t.attempt,
                "precision": t.precision,
                "runtime_ratio": t.runtime_ratio,
                "time_before": t.time_before,
                "time_after": t.time_after,
                "compile_log_path": t.compile_log_path,
                "run_log_path": t.run_log_path,
                "error_history": t.error_history,
            }
        )
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = []
    total = len(tasks)
    success = sum(1 for t in tasks if t.status == "success")
    failed = sum(1 for t in tasks if t.status == "failed")
    lines.append("=" * 80)
    lines.append("CUDA Eval Pipeline Report")
    lines.append("=" * 80)
    lines.append(f"total={total} success={success} failed={failed}")
    lines.append("-" * 80)
    for t in tasks:
        lines.append(
            f"{t.task_dir_name:70} status={t.status:8} attempt={t.attempt} "
            f"precision={t.precision or '-'} ratio={t.runtime_ratio or '-'}"
        )
        if t.error_history:
            last = t.error_history[-1]
            lines.append(
                f"  last_error[{last.get('type')}]: {str(last.get('message', ''))[:300]}"
            )
    lines.append("-" * 80)
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, txt_path


def task_to_dict(task: TaskState) -> dict[str, Any]:
    return asdict(task)


def task_from_dict(data: dict[str, Any]) -> TaskState:
    return TaskState(
        level=data.get("level", ""),
        problem_id=int(data.get("problem_id", 0)),
        problem_name=data.get("problem_name", ""),
        task_dir_name=data.get("task_dir_name", ""),
        status=data.get("status", "pending"),
        attempt=int(data.get("attempt", 0)),
        generated_code=data.get("generated_code", ""),
        error_history=list(data.get("error_history", [])),
        precision=data.get("precision", ""),
        runtime_ratio=data.get("runtime_ratio", ""),
        time_before=data.get("time_before", ""),
        time_after=data.get("time_after", ""),
        compile_log_path=data.get("compile_log_path", ""),
        run_log_path=data.get("run_log_path", ""),
    )


def save_checkpoint(
    report_base: Path, args, tasks: list[TaskState], iter_idx: int, note: str = ""
):
    checkpoint_path = report_base / "checkpoint.json"
    payload = {
        "meta": {
            "saved_at": datetime.now().isoformat(),
            "iter_idx": iter_idx,
            "note": note,
            "run_name": args.run_name,
            "model": args.model,
            "level": args.level,
            "max_iter": args.max_iter,
            "kernelbench_namespace": args.kernelbench_namespace,
        },
        "tasks": [task_to_dict(t) for t in tasks],
    }
    checkpoint_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_resume_tasks(resume_dir: Path) -> tuple[list[TaskState], int]:
    checkpoint_path = resume_dir / "checkpoint.json"
    report_path = resume_dir / "pipeline_report.json"

    if checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        tasks = [task_from_dict(d) for d in payload.get("tasks", [])]
        start_iter = int(payload.get("meta", {}).get("iter_idx", 0)) + 1
        return tasks, start_iter

    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        tasks = [task_from_dict(d) for d in payload]
        max_attempt = max((t.attempt for t in tasks), default=0)
        return tasks, max_attempt + 1

    raise FileNotFoundError(
        f"Resume directory has neither checkpoint.json nor pipeline_report.json: {resume_dir}"
    )


def run_task_pipeline(
    args,
    task: TaskState,
    common_h: Path,
    kernelbench_base: Path,
    gpu_pool: GpuPool,
    logs_dir: Path,
    llm_trace_dir: Path | None,
) -> TaskState:
    if task.status != "pending":
        return task

    start_attempt = max(1, task.attempt + 1)
    for attempt in range(start_attempt, args.max_iter + 1):
        task.attempt = attempt

        ok = generate_for_task(args, task, llm_trace_dir)
        if not ok:
            if (
                task.error_history
                and task.error_history[-1].get("type") == "kernel_missing"
            ):
                task.status = "failed"
                return task
            continue

        try:
            write_cuda_eval_code(args, task)
            task_base = convert_to_task_dir(args, task, common_h, kernelbench_base)
        except Exception as e:
            task.error_history.append(
                {"attempt": task.attempt, "type": "convert", "message": str(e)}
            )
            continue

        run_ok = compile_and_run(args, task, task_base, gpu_pool, logs_dir)
        if run_ok:
            task.status = "success"
            return task

    task.status = "failed"
    return task


def main():
    args = parse_args()
    common_h = find_common_h(args.common_h)
    namespace = derive_kernelbench_namespace(args.run_name, args.kernelbench_namespace)
    kernelbench_base = KERNELBENCH_TASK_ROOT / namespace
    kernelbench_base.mkdir(parents=True, exist_ok=True)

    if args.num_gpus <= 0:
        args.num_gpus = detect_num_gpus()
    print(f"use num_gpus={args.num_gpus}")
    print(f"kernelbench output root: {kernelbench_base}")

    start_iter = 1
    if args.resume:
        report_base = Path(args.resume).resolve()
        if not report_base.exists():
            raise FileNotFoundError(f"--resume path does not exist: {report_base}")
        tasks, start_iter = load_resume_tasks(report_base)
        print(f"resume mode: {report_base}")
        print(f"resumed tasks: {len(tasks)}, start_iter={start_iter}")
    else:
        tasks = collect_tasks(args)
        if not tasks:
            print("No tasks collected. Exit.")
            return
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_base = Path(args.report_dir) / f"{args.run_name}_{now}"

    logs_dir = report_base / "logs"
    llm_trace_dir = (
        None if args.disable_llm_trace else (report_base / args.llm_trace_subdir)
    )
    logs_dir.mkdir(parents=True, exist_ok=True)
    if llm_trace_dir is not None:
        llm_trace_dir.mkdir(parents=True, exist_ok=True)

    gpu_pool = GpuPool(args.num_gpus)
    save_checkpoint(report_base, args, tasks, max(0, start_iter - 1), note="startup")

    pending_tasks = [t for t in tasks if t.status == "pending"]
    if not pending_tasks:
        print("No pending tasks to run.")
    else:
        task_workers = max(1, args.parallel_gen, args.compile_workers)
        print(
            f"task pipeline mode: workers={task_workers}, "
            f"pending={len(pending_tasks)}, max_iter_per_task={args.max_iter}"
        )
        completed = 0
        with ThreadPoolExecutor(max_workers=task_workers) as pool:
            future_map = {
                pool.submit(
                    run_task_pipeline,
                    args,
                    task,
                    common_h,
                    kernelbench_base,
                    gpu_pool,
                    logs_dir,
                    llm_trace_dir,
                ): task
                for task in pending_tasks
            }
            progress = tqdm(
                as_completed(future_map), total=len(future_map), desc="tasks pipeline"
            )
            for fut in progress:
                task = future_map[fut]
                try:
                    _ = fut.result()
                except Exception as e:
                    task.error_history.append(
                        {
                            "attempt": task.attempt,
                            "type": "pipeline",
                            "message": f"exception: {e}",
                        }
                    )
                    task.status = "failed"

                completed += 1
                succ = sum(1 for t in tasks if t.status == "success")
                fail = sum(1 for t in tasks if t.status == "failed")
                pend = sum(1 for t in tasks if t.status == "pending")
                progress.set_postfix(
                    {
                        "done": completed,
                        "success": succ,
                        "failed": fail,
                        "pending": pend,
                    }
                )
                save_checkpoint(
                    report_base,
                    args,
                    tasks,
                    max((t.attempt for t in tasks), default=0),
                    note="task_completed",
                )

    json_path, txt_path = write_report(tasks, report_base)
    save_checkpoint(
        report_base,
        args,
        tasks,
        max((t.attempt for t in tasks), default=0),
        note="finished",
    )
    print(f"\nreport json: {json_path}")
    print(f"report txt : {txt_path}")


if __name__ == "__main__":
    main()
