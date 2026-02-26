"""
多轮对话批量生成所有 KernelBench 级别的算子。

工作流程：生成 -> 执行/评估 -> 反馈 -> 优化，进行多轮迭代。
支持在模型达到 token 限制时重新开始对话（finish_reason="length"）。
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional, Callable
from types import SimpleNamespace

import pydra

# 设置环境变量（必须在任何 PyTorch 操作之前）
# 使用独立的编译目录避免锁冲突
os.environ["TORCH_EXTENSIONS_DIR"] = f"/tmp/torch_extensions_{os.environ.get('USER', 'default')}"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# KernelBench 导入
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string, KernelExecResult
from kernelbench.kernel_static_checker import validate_kernel_static
from kernelbench.prompt_constructor_toml import get_custom_prompt, get_prompt_for_backend
from kernelbench.utils import (
    SERVER_PRESETS,
    query_server,
    extract_first_code,
    maybe_multithread,
)

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cleanup_cuda_memory(verbose: bool = False):
    """清理 CUDA 内存和垃圾回收。"""
    try:
        import torch
        if torch.cuda.is_available():
            if verbose:
                alloc_before = torch.cuda.memory_allocated() / 1024**2
                reserved_before = torch.cuda.memory_reserved() / 1024**2
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            if verbose:
                alloc_after = torch.cuda.memory_allocated() / 1024**2
                reserved_after = torch.cuda.memory_reserved() / 1024**2
                print(f"[Memory] CUDA 内存: 已分配 {alloc_before:.1f}MB -> {alloc_after:.1f}MB, "
                      f"预留 {reserved_before:.1f}MB -> {reserved_after:.1f}MB")
    except Exception:
        pass
    gc.collect()


def setup_compile_environment(max_jobs: int = 1):
    """
    设置 CUDA 编译环境变量以限制内存使用。
    
    PyTorch 的 cpp_extension 在编译 CUDA 内核时会使用多进程并行编译。
    默认使用所有 CPU 核心，这可能导致内存不足。
    
    参数：
        max_jobs: 编译时的最大并行作业数（默认 1，设为 0 则使用所有核心）
    """
    import os
    
    # 设置最大并行编译作业数
    # 这是最重要的设置，控制并行编译进程数
    os.environ["MAX_JOBS"] = str(max_jobs)
    
    # 减少 nvcc 线程数（在 MAX_JOBS 内的每个作业中）
    # 这进一步限制内存使用
    current_nvcc_flags = os.environ.get("NVCC_APPEND_FLAGS", "")
    if "--threads" not in current_nvcc_flags:
        os.environ["NVCC_APPEND_FLAGS"] = current_nvcc_flags + " --threads 1"


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class WorkArgs:
    """单个问题/样本组合的工作项。"""
    problem_id: int
    sample_id: int


@dataclass
class AttemptRecord:
    """单次生成尝试的记录。"""
    kernel_code: str
    summary: Optional[str]
    eval_result: KernelExecResult


# =============================================================================
# 配置类
# =============================================================================

class GenerationMultiturnAllLevelsConfig(pydra.Config):
    """多轮生成所有级别的配置。"""

    def __init__(self):
        # 数据集设置
        self.dataset_src = pydra.REQUIRED
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = "all"
        self.include_level4_expand = False

        # 问题子集
        self.subset = (None, None)  # (start_id, end_id)
        self.problem_subset_file = None

        # 运行设置
        self.run_name = pydra.REQUIRED
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")
        self.store_type = "local"

        # 采样设置
        self.num_samples = 1
        self.num_workers = 1
        self.api_query_interval = 0.0

        # 多轮设置
        self.max_turns = 4
        self.early_stop_on_correct = True
        self.max_conversation_segments = 3  # 因长度限制重启对话的最大次数
        
        # 内存优化设置
        self.cuda_memory_cleanup = True  # 是否在每轮后清理 CUDA 内存
        self.cuda_compile_max_jobs = 1   # CUDA 编译时的最大并行作业数（默认 1，设为 0 则使用所有核心）
        
        # 输入缩放设置（用于内存不足时减小输入尺寸）
        self.scale_down_inputs = False   # 是否缩小输入尺寸以节省内存
        self.scale_max_batch = 2         # 缩小后的最大 batch size
        self.scale_max_channels = 8      # 缩小后的最大 channels
        self.scale_max_spatial = 256     # 缩小后的最大空间维度（高/宽）

        # 推理设置
        self.server_type = None
        self.model_name = None
        self.max_tokens = None
        self.temperature = 0.0
        self.is_reasoning_model = False
        self.reasoning_effort = "low"
        self.budget_tokens = 0

        # 后端设置
        self.backend = "cuda"
        self.precision = "fp32"
        self.prompt_option = "one_shot"
        self.include_hardware_info = False
        self.hardware_gpu_name = None
        self.custom_prompt_key = None

        # 日志设置
        self.verbose = False
        self.log_prompt = False
        self.log_conversation = True
        self.conversation_system_prompt = ""
        self.check_kernel = True

    def to_dict(self):
        """将配置转换为字典以便序列化。"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self):
        return f"GenerationMultiturnAllLevelsConfig({self.to_dict()})"


# =============================================================================
# 输入缩放工具
# =============================================================================

def scale_down_inputs(
    input_args: list, 
    max_batch: int = 2, 
    max_channels: int = 8, 
    max_spatial: int = 256
) -> list:
    """
    减小输入张量尺寸以避免 OOM。
    
    例如将 [16, 64, 2048, 2048] 减小为 [2, 8, 256, 256]
    
    参数：
        input_args: 输入参数列表
        max_batch: 最大 batch size
        max_channels: 最大 channels
        max_spatial: 最大空间维度（高/宽）
    
    返回：
        缩放后的输入参数列表
    """
    import torch
    scaled_inputs = []
    
    for x in input_args:
        if isinstance(x, torch.Tensor):
            if x.dim() == 4:
                # 4D 张量 (batch, channels, height, width)
                b = min(x.size(0), max_batch)
                c = min(x.size(1), max_channels)
                h = min(x.size(2), max_spatial)
                w = min(x.size(3), max_spatial)
                x = x[:b, :c, :h, :w]
            elif x.dim() == 3:
                # 3D 张量
                d0 = min(x.size(0), max_batch)
                d1 = min(x.size(1), max_channels)
                d2 = min(x.size(2), max_spatial)
                x = x[:d0, :d1, :d2]
            elif x.dim() == 2:
                # 2D 张量
                d0 = min(x.size(0), max_spatial)
                d1 = min(x.size(1), max_spatial)
                x = x[:d0, :d1]
            scaled_inputs.append(x)
        else:
            scaled_inputs.append(x)
    
    return scaled_inputs


# =============================================================================
# 级别规格
# =============================================================================

def get_all_level_specs(
    dataset_src: str, include_level4_expand: bool
) -> list[tuple[int, Optional[str], str]]:
    """
    获取要运行的所有级别列表。
    
    返回：
        (dataset_level, local_subdir, level_label) 元组列表。
        level_label 用于文件命名（例如 "1", "4", "level4_expand"）。
    """
    specs: list[tuple[int, Optional[str], str]] = [
        (1, None, "1"),
        (2, None, "2"),
        (3, None, "3"),
        (4, None, "4"),
    ]
    if include_level4_expand and dataset_src == "local":
        specs.append((4, "level4_expand", "level4_expand"))
    elif include_level4_expand and dataset_src != "local":
        raise ValueError("level4_expand 仅支持 dataset_src=local")
    return specs


def parse_problem_subset_file(path: str) -> dict[str, set[int]]:
    """
    解析问题子集文件。每行格式：level_<level>_problem_<id>
    
    返回：
        将 level_label 映射到问题 ID 集合的字典。
    """
    pattern = re.compile(r"level_(level\d|level4_expand)_problem_(\d+)")
    level_to_ids: dict[str, set[int]] = {}
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = pattern.match(line)
            if not m:
                continue
            level_key, pid_str = m.group(1), m.group(2)
            level_label = "level4_expand" if level_key == "level4_expand" else level_key.replace("level", "")
            pid = int(pid_str)
            level_to_ids.setdefault(level_label, set()).add(pid)
    
    return level_to_ids


# =============================================================================
# 文件工具
# =============================================================================

def check_kernel_exists(run_dir: str, level_label: str, problem_id: int, sample_id: int) -> bool:
    """检查内核文件是否已存在。"""
    kernel_path = os.path.join(
        run_dir, f"level_{level_label}_problem_{problem_id}_sample_{sample_id}_kernel.py"
    )
    return os.path.exists(kernel_path)


def run_dir_has_existing_kernels(run_dir: str) -> bool:
    """检查运行目录是否有任何生成的内核文件。"""
    if not os.path.isdir(run_dir):
        return False
    for name in os.listdir(run_dir):
        if name.endswith("_kernel.py") and name.startswith("level_"):
            return True
    return False


def ask_resume(run_dir: str) -> bool:
    """
    询问用户是继续（跳过现有文件）还是重新开始。
    在非交互模式下，默认为继续。
    """
    if not sys.stdin.isatty():
        print(
            f"运行目录已存在且包含内核文件：{run_dir}。"
            "非交互模式：继续（跳过现有，生成缺失的）。"
        )
        return True
    
    print(f"\n⚠️  运行目录已存在：{run_dir}")
    print("   之前的运行可能已中断。如果继续，将跳过现有内核。")
    
    while True:
        try:
            answer = input("   继续（跳过现有，仅生成缺失的）？[Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n已中止。")
            sys.exit(1)
        if answer in ("", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("   请输入 Y 或 n。")


# =============================================================================
# 带 Finish Reason 的推理服务器
# =============================================================================

def create_inference_server_with_metadata(
    server_type: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    verbose: bool = False,
    is_reasoning_model: bool = False,
    reasoning_effort: str = "low",
    budget_tokens: int = 0,
) -> Callable:
    """
    创建返回内容和 finish_reason 的推理服务器。
    
    返回一个可调用对象，接受消息并返回包含以下内容的 SimpleNamespace：
        - content: str
        - finish_reason: str | None
    """
    def _query_llm(messages: list[dict] | str) -> SimpleNamespace:
        # 从预设构建服务器参数
        if server_type and server_type in SERVER_PRESETS:
            server_args = SERVER_PRESETS[server_type].copy()
        else:
            server_args = {}
        
        # 用显式参数覆盖
        if model_name is not None and model_name != "None":
            server_args["model_name"] = model_name
        if max_tokens is not None and max_tokens != "None":
            server_args["max_tokens"] = max_tokens
        if temperature is not None and temperature != "None":
            server_args["temperature"] = temperature
        if reasoning_effort is not None:
            server_args["reasoning_effort"] = reasoning_effort
        if budget_tokens is not None:
            server_args["budget_tokens"] = budget_tokens
        
        server_args["is_reasoning_model"] = is_reasoning_model
        server_args["verbose"] = verbose
        
        # 我们需要修改 query_server 以返回 finish_reason
        # 现在包装它并使用自定义实现
        return _query_server_with_finish_reason(messages, server_type, **server_args)
    
    return _query_llm


def _extract_reasoning_content(response, is_reasoning_model: bool) -> str | None:
    """
    从响应中提取 reasoning_content。
    
    支持 OpenAI o1/o3 (reasoning_content) 和 Anthropic Claude (thinking)。
    """
    if not is_reasoning_model:
        return None
    
    try:
        message = response.choices[0].message
        
        # OpenAI o1/o3 风格
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            return message.reasoning_content
        
        # Anthropic Claude 风格 (thinking 字段)
        if hasattr(message, "thinking") and message.thinking:
            if isinstance(message.thinking, dict):
                return message.thinking.get("content", "")
            return str(message.thinking)
        
        # 检查 provider_specific_fields
        if hasattr(message, "provider_specific_fields"):
            fields = message.provider_specific_fields
            if isinstance(fields, dict):
                return fields.get("reasoning_content") or fields.get("thinking")
        
        return None
    except Exception:
        return None


def _query_server_with_finish_reason(
    prompt: list[dict] | str,
    server_type: str | None = None,
    **kwargs
) -> SimpleNamespace:
    """
    查询服务器并返回内容、finish_reason 和 reasoning_content。
    
    这是保留 finish_reason 和 reasoning_content 的 query_server 的修改版本。
    """
    import litellm
    from openai import OpenAI
    
    # 获取参数
    model_name = kwargs.get("model_name", "default")
    temperature = float(kwargs.get("temperature", 0.7))
    max_tokens = int(kwargs.get("max_tokens", 4096))
    top_p = float(kwargs.get("top_p", 0.9))
    top_k = int(kwargs.get("top_k", 40))
    num_completions = int(kwargs.get("num_completions", 1))
    system_prompt = kwargs.get("system_prompt", "")
    is_reasoning_model = kwargs.get("is_reasoning_model", False)
    reasoning_effort = kwargs.get("reasoning_effort", "low")
    budget_tokens = int(kwargs.get("budget_tokens", 0))
    verbose = kwargs.get("verbose", False)
    
    # 本地服务器处理
    if server_type == "local":
        server_address = kwargs.get("server_address", "localhost")
        server_port = kwargs.get("server_port", 10210)
        url = f"http://{server_address}:{server_port}"
        
        default_headers = None
        if os.environ.get("LOCAL_SERVER_HOST"):
            default_headers = {"Host": os.environ.get("LOCAL_SERVER_HOST")}
        
        client = OpenAI(
            api_key=os.environ.get("SGLANG_KEY", "None"),
            base_url=f"{url}/v1",
            timeout=None,
            max_retries=0,
            default_headers=default_headers,
        )
        
        if isinstance(prompt, str):
            response = client.completions.create(
                model="default",
                prompt=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            content = response.choices[0].text
            finish_reason = response.choices[0].finish_reason
            reasoning_content = None
        else:
            response = client.chat.completions.create(
                model="default",
                messages=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            reasoning_content = _extract_reasoning_content(response, is_reasoning_model)
        
        return SimpleNamespace(
            content=content or "", 
            finish_reason=finish_reason,
            reasoning_content=reasoning_content
        )
    
    # 其他提供者的 LiteLLM 处理
    messages = []
    if isinstance(prompt, list) and prompt and prompt[0].get("role") == "system":
        messages = prompt
    else:
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.extend(prompt)
    
    completion_kwargs = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "n": num_completions,
    }
    
    if is_reasoning_model:
        if reasoning_effort and ("openai" in model_name.lower() or "o1" in model_name or "o3" in model_name):
            completion_kwargs["reasoning_effort"] = reasoning_effort
        if budget_tokens > 0 and "anthropic" in model_name.lower():
            completion_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
    else:
        completion_kwargs["temperature"] = temperature
        completion_kwargs["top_p"] = top_p
        if "openai/" not in model_name.lower() and "gpt" not in model_name.lower():
            completion_kwargs["top_k"] = top_k
    
    # 如果需要，添加托管 vLLM 参数（仅对 hosted_vllm 模型）
    if model_name.startswith("hosted_vllm/"):
        api_base = os.environ.get("HOSTED_VLLM_API_BASE", "")
        host = os.environ.get("HOSTED_VLLM_HOST", "")
        if api_base:
            completion_kwargs["api_base"] = api_base
        if host:
            completion_kwargs["extra_headers"] = {"Host": host}
    
    litellm.drop_params = True
    try:
        response = litellm.completion(**completion_kwargs)
    except litellm.ContextWindowExceededError as e:
        # 上下文窗口超限，返回特殊标记让调用方处理
        if verbose:
            print(f"[LLM] ContextWindowExceededError: {e}")
        return SimpleNamespace(
            content="",
            finish_reason="context_window_exceeded",
            reasoning_content=None,
            error=str(e)
        )
    except Exception as e:
        # 其他错误也返回特殊标记
        if verbose:
            print(f"[LLM] Error: {type(e).__name__}: {e}")
        return SimpleNamespace(
            content="",
            finish_reason="error",
            reasoning_content=None,
            error=str(e)
        )
    
    content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    reasoning_content = _extract_reasoning_content(response, is_reasoning_model)
    
    # 注意：即使 content 为 None，也返回结果让调用方处理（特别是处理 length 限制的情况）
    return SimpleNamespace(
        content=content or "", 
        finish_reason=finish_reason,
        reasoning_content=reasoning_content
    )


# =============================================================================
# 文本提取
# =============================================================================

def extract_summary_after_first_codeblock(raw: str) -> Optional[str]:
    """提取第一个代码块后的尾部文本作为摘要。"""
    if not raw:
        return None
    
    s = raw.strip()
    start = s.find("```")
    if start < 0:
        return None
    
    end = s.find("```", start + 3)
    if end < 0:
        return None
    
    end2 = s.find("```", end + 3)
    boundary = end2 if end2 >= 0 else end + 3
    tail = s[boundary:].strip()
    
    return tail[:1200] if tail else None


# =============================================================================
# 评估工具
# =============================================================================

def make_parsing_failure_result(message: str) -> KernelExecResult:
    """为解析失败创建 KernelExecResult。"""
    return KernelExecResult(
        compiled=False,
        correctness=False,
        metadata={"parsing_error": message}
    )


def eval_result_to_turn_metric(turn_index: int, segment_index: int, result: KernelExecResult) -> dict:
    """将 KernelExecResult 转换为轮次指标字典。"""
    md = result.metadata or {}
    
    speedup = None
    ref_runtime_us = None
    runtime_us = None
    
    if result.runtime > 0 and result.ref_runtime > 0:
        runtime_us = round(result.runtime, 4)
        ref_runtime_us = round(result.ref_runtime, 4)
        speedup = round(float(result.ref_runtime) / float(result.runtime), 4)
    
    error = None
    if not result.compiled:
        error = str(md.get("compilation_error") or md.get("parsing_error") or md.get("compilation_error_name", "Unknown"))
    elif not result.correctness:
        error = str(md.get("runtime_error") or md.get("correctness_issue") or "Unknown")
    
    return {
        "turn": turn_index,
        "segment": segment_index,
        "compiled": result.compiled,
        "correctness": result.correctness,
        "speedup": speedup,
        "ref_runtime_us": ref_runtime_us,
        "runtime_us": runtime_us,
        "error": error,
    }


def turn_metric_to_eval_result(metric: dict) -> KernelExecResult:
    """从轮次指标恢复 KernelExecResult（用于恢复）。"""
    runtime = metric.get("runtime_us")
    ref_runtime = metric.get("ref_runtime_us")
    
    return KernelExecResult(
        compiled=bool(metric.get("compiled", False)),
        correctness=bool(metric.get("correctness", False)),
        runtime=float(runtime) if runtime is not None else -1.0,
        ref_runtime=float(ref_runtime) if ref_runtime is not None else -1.0,
        metadata={"error": metric.get("error")} if metric.get("error") else {},
    )


def eval_with_retries(
    ref_arch_src: str,
    kernel_src: str,
    config: GenerationMultiturnAllLevelsConfig,
    max_retries: int = 3,
) -> KernelExecResult:
    """
    评估内核，对瞬态编译锁定错误进行重试。
    
    如果 config.scale_down_inputs 为 True，会自动缩小输入尺寸以节省内存。
    """
    last_err: Optional[str] = None
    
    # 设置输入缩放环境变量（如果启用）
    if getattr(config, "scale_down_inputs", False):
        os.environ["KERNELBENCH_SCALE_DOWN_INPUTS"] = "1"
        os.environ["KERNELBENCH_SCALE_MAX_BATCH"] = str(getattr(config, "scale_max_batch", 2))
        os.environ["KERNELBENCH_SCALE_MAX_CHANNELS"] = str(getattr(config, "scale_max_channels", 8))
        os.environ["KERNELBENCH_SCALE_MAX_SPATIAL"] = str(getattr(config, "scale_max_spatial", 256))
        if config.verbose:
            print(f"[Eval] 启用输入缩放: batch={config.scale_max_batch}, "
                  f"channels={config.scale_max_channels}, spatial={config.scale_max_spatial}")
    else:
        # 确保禁用输入缩放
        os.environ.pop("KERNELBENCH_SCALE_DOWN_INPUTS", None)
    
    for i in range(max_retries):
        try:
            result = eval_kernel_against_ref(
                ref_arch_src,
                kernel_src,
                verbose=config.verbose,
                measure_performance=True,
                timing_method=getattr(config, "timing_method", "cuda_event"),
                backend=config.backend,
                precision=get_torch_dtype_from_string(config.precision),
            )
            if result is not None:
                return result
            last_err = "评估返回 None（可能是编译锁定）。请重试。"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        finally:
            # 清理 CUDA 内存
            cleanup_cuda_memory()
        time.sleep(1.0 * (i + 1))
    
    return make_parsing_failure_result(last_err or "评估因未知错误失败。")


# =============================================================================
# 错误消息处理
# =============================================================================

def extract_cuda_error_details(error) -> tuple[str, str]:
    """
    从 CUDA 错误中提取错误类型和详细信息。
    
    返回：
        (错误类型描述, 详细错误信息)
    """
    # 确保 error 是字符串
    if isinstance(error, Exception):
        error = str(error)
    if not isinstance(error, str):
        return "CUDA 错误", "未知错误"
    error_lower = error.lower()
    
    # CUDA 错误类型映射
    cuda_errors = {
        "cudaerrorillegaladdress": ("CUDA 非法内存访问", [
            "an illegal memory access was encountered",
            "illegal memory access"
        ]),
        "cudaerroroutofmemory": ("CUDA 内存不足", [
            "out of memory"
        ]),
        "cudaerrormisalignedaddress": ("CUDA 未对齐地址访问", [
            "misaligned address"
        ]),
        "cudaerrorlaunchfailure": ("CUDA 启动失败", [
            "launch failure"
        ]),
        "cudaerrorinvalidconfiguration": ("CUDA 配置无效", [
            "invalid configuration",
            "too many resources requested"
        ]),
        "cudaerrordevicesync": ("CUDA 设备同步错误", [
            "device-side assert triggered"
        ]),
    }
    
    for error_code, (error_type, patterns) in cuda_errors.items():
        if error_code in error_lower or any(p in error_lower for p in patterns):
            return error_type, error
    
    return "CUDA 错误", error


def extract_compilation_error(error) -> str:
    """
    从 PyTorch load_inline 编译错误中提取有用的 nvcc 错误信息。
    
    过滤掉 Python 回溯，保留实际的编译错误。
    """
    # 确保 error 是字符串
    if isinstance(error, Exception):
        error = str(error)
    if not error or not isinstance(error, str):
        return "未知编译错误"
    
    # nvcc 错误行模式
    nvcc_patterns = [
        r'[\w\/\.]+\.(cu|cpp|h)\(\d+\):\s*error:\s*.+',  # 文件(行): error: 消息
        r'[\w\/\.]+\.(cu|cpp|h):\d+:\d+:\s*error:\s*.+',  # 文件:行:列: error: 消息
        r'error:\s*.+',  # 一般 error: 消息
        r'\d+\s+errors?\s+detected',  # X errors detected
        r'nvcc\s+fatal',  # nvcc fatal
    ]
    
    lines = error.split('\n')
    relevant_lines = []
    
    # 查找有用的错误行
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # 跳过 Python 回溯行
        if any(skip in line for skip in [
            'File "', 'Traceback', 'line', 'in <module>', 
            'torch.utils.cpp_extension', 'subprocess.CalledProcessError'
        ]):
            continue
        
        # 收集 nvcc 错误行
        for pattern in nvcc_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # 包含这行和可能的上下文
                relevant_lines.append(line_stripped)
                # 也包含下一行（可能有额外信息）
                if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith('File '):
                    relevant_lines.append(lines[i + 1].strip())
                break
    
    if relevant_lines:
        # 去重并保持顺序
        seen = set()
        unique_lines = []
        for line in relevant_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        return '\n'.join(unique_lines[:20])  # 限制行数
    
    # 如果没有找到 nvcc 错误，返回清理后的原始错误
    # 去除 Python 回溯
    cleaned_lines = []
    for line in lines:
        if any(skip in line for skip in ['File "', 'Traceback', '  File ']):
            continue
        if line.strip():
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines[:30])  # 限制行数
    return result if result else error[:1000]


def is_async_cuda_error_indicator(error) -> tuple[bool, str]:
    """
    检测错误是否是异步 CUDA 错误的表现（在 PyTorch 调用处触发）。
    
    这些错误通常发生在：
    - torch.cuda.synchronize()
    - torch.manual_seed()
    - torch.cuda.manual_seed()
    - torch.cuda.empty_cache()
    - 其他 PyTorch CUDA 同步点
    
    返回：
        (是否是异步错误, 错误类型描述)
    """
    # 确保 error 是字符串
    if isinstance(error, Exception):
        error = str(error)
    if not isinstance(error, str):
        return False, ""
    error_lower = error.lower()
    
    # 检查是否包含 CUDA 错误关键词
    has_cuda_error = any(cuda_err in error_lower for cuda_err in [
        'cudaerror', 'cuda error', 'an illegal memory access',
        'device-side assert', 'out of memory', 'misaligned address'
    ])
    
    if not has_cuda_error:
        return False, ""
    
    # 检查是否在 PyTorch 同步点触发
    async_indicators = [
        ('torch.cuda.synchronize', 'torch.cuda.synchronize()'),
        ('torch.manual_seed', 'torch.manual_seed()'),
        ('torch.cuda.manual_seed', 'torch.cuda.manual_seed()'),
        ('torch.cuda.empty_cache', 'torch.cuda.empty_cache()'),
        ('torch.cuda.init', 'torch.cuda.init()'),
        ('cudastream', 'CUDA Stream 操作'),
        ('cuda event', 'CUDA Event 操作'),
    ]
    
    for indicator, description in async_indicators:
        if indicator in error_lower:
            return True, description
    
    return False, ""


def extract_cuda_kernel_error_location(error) -> str | None:
    """
    尝试从错误中提取 CUDA 内核代码的具体错误位置。
    
    如果找到具体的 CUDA 代码行号或内核名称，返回该信息。
    否则返回 None。
    """
    # 确保 error 是字符串
    if isinstance(error, Exception):
        error = str(error)
    if not isinstance(error, str):
        return None
    lines = error.split('\n')
    
    # 查找 CUDA 内核相关的代码位置
    kernel_patterns = [
        r'at\s+([\w\/\.]+\.cu):(\d+)',  # at file.cu:line
        r'([\w\/\.]+\.cu)\((\d+)\)',     # file.cu(line)
        r'kernel\s*<<[^>]+>>',            # kernel<<<...>>>
        r'__global__.*\{',                # __global__ 函数
        r'CUDA KERNEL\s*:\s*(.+)',        # CUDA KERNEL: 消息
    ]
    
    locations = []
    for line in lines:
        line_stripped = line.strip()
        for pattern in kernel_patterns:
            match = re.search(pattern, line_stripped, re.IGNORECASE)
            if match:
                locations.append(line_stripped)
                break
    
    if locations:
        return '\n'.join(locations[:5])
    
    return None


def extract_runtime_error(error) -> str:
    """
    从运行时错误中提取有用的 CUDA 错误信息。
    
    特别处理两种情况：
    1. 异步 CUDA 错误（在 PyTorch 调用处触发）- 返回简洁提示
    2. 真正的 CUDA 内核代码错误 - 返回详细位置和错误信息
    """
    # 确保 error 是字符串
    if isinstance(error, Exception):
        error = str(error)
    if not error or not isinstance(error, str):
        return "未知运行时错误"
    
    # 首先检查是否是 CUDA 错误
    error_type, full_error = extract_cuda_error_details(error)
    
    # 检查是否是异步 CUDA 错误（在 PyTorch 同步点触发）
    is_async, trigger_point = is_async_cuda_error_indicator(error)
    
    if is_async:
        # 尝试查找具体的 CUDA 内核错误位置
        kernel_location = extract_cuda_kernel_error_location(error)
        
        if kernel_location:
            # 找到了具体的 CUDA 代码位置，返回详细信息
            return (
                f"错误类型: {error_type}\n"
                f"CUDA 内核错误位置:\n{kernel_location}\n"
                f"注意：此错误在 {trigger_point} 处被检测到，"
                f"可能是之前执行的 CUDA 内核中的异步错误。"
            )
        else:
            # 没有找到具体位置，返回简洁提示
            return (
                f"错误类型: {error_type}\n"
                f"此错误在 {trigger_point} 处被检测到，"
                f"表明您之前生成的 CUDA 内核代码存在问题。\n"
                f"常见原因：\n"
                f"- 越界内存访问（数组索引超出范围）\n"
                f"- 未初始化的共享内存使用\n"
                f"- 线程同步问题（__syncthreads 使用不当）\n"
                f"- 无效的指针解引用\n"
                f"请仔细检查您的 CUDA 内核实现。"
            )
    
    # 非异步错误，正常提取 CUDA 错误信息
    lines = error.split('\n')
    relevant_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # 跳过 Python 回溯
        if any(skip in line for skip in [
            'File "', 'Traceback', '  File ', 'line ', ', in ',
            'torch._C', 'torch.cuda', 'RuntimeError:'
        ]):
            # 但保留 RuntimeError 的内容
            if 'RuntimeError:' in line and len(line_stripped) > 20:
                parts = line.split('RuntimeError:', 1)
                if len(parts) > 1 and parts[1].strip():
                    relevant_lines.append(parts[1].strip())
            continue
        
        # 保留 CUDA 错误信息
        line_lower = line.lower()
        if any(cuda_key in line_lower for cuda_key in [
            'cuda', 'cudastream', 'cudaevent', 'cudakernel'
        ]):
            relevant_lines.append(line_stripped)
    
    # 构建结果
    result_parts = []
    
    if error_type != "CUDA 错误":
        result_parts.append(f"错误类型: {error_type}")
    
    if relevant_lines:
        result_parts.extend(relevant_lines[:15])
    
    if not result_parts:
        non_traceback = []
        for line in lines:
            if not any(skip in line for skip in ['File "', 'Traceback', '  File ']):
                if line.strip():
                    non_traceback.append(line.strip())
        if non_traceback:
            return '\n'.join(non_traceback[:20])
        return error[:800]
    
    return '\n'.join(result_parts)


def sanitize_error_message(error, error_type: str = "general") -> str:
    """
    清理错误消息，保留技术细节但去除 Python 回溯噪音。
    
    参数：
        error: 原始错误消息（字符串或异常对象）
        error_type: 错误类型 ("compilation", "runtime", "general")
    
    返回：
        清理后的错误消息，保留有用的技术细节
    """
    # 确保 error 是字符串
    if isinstance(error, Exception):
        error = str(error)
    if not error or not isinstance(error, str):
        return "未知错误"
    
    # 根据错误类型使用专门的提取函数
    if error_type == "compilation":
        return extract_compilation_error(error)
    elif error_type == "runtime":
        return extract_runtime_error(error)
    
    # 一般错误处理
    error_lower = error.lower()
    
    # 检查是否是 CUDA 错误
    cuda_error_type, _ = extract_cuda_error_details(error)
    if cuda_error_type != "CUDA 错误":
        # 是已知的 CUDA 错误，返回详细信息
        return extract_runtime_error(error)
    
    # 保留编译错误原样
    if any(pattern in error_lower for pattern in [
        "error: ",
        "syntax error",
        "undeclared identifier",
        "undefined reference",
    ]):
        return extract_compilation_error(error)
    
    # 对于其他错误，如果太长则截断
    if len(error) > 800:
        return error[:800] + "\n...（错误消息已截断）"
    
    return error


# =============================================================================
# 反馈格式化
# =============================================================================

def format_feedback_from_eval_result(result: KernelExecResult) -> str:
    """
    从评估结果格式化反馈消息。
    返回详细的错误信息，保留编译和运行时错误的技术细节。
    """
    md = result.metadata or {}
    
    # 解析错误
    if md.get("parsing_error"):
        return (
            "您之前的回答由于不符合所需格式而无法解析。"
            "以下是错误信息：\n"
            f"{md.get('parsing_error')}"
        )
    
    # 编译错误 - 使用专门的编译错误提取
    if not result.compiled:
        err = md.get("compilation_error")
        if err is None:
            err = md.get("compilation_error_name", "未知编译错误")
        # 提取编译错误详情（nvcc 错误信息）
        err = sanitize_error_message(err, error_type="compilation")
        separator = "=" * 50
        return (
            "您之前的回答编译失败。以下是详细的编译错误信息：\n" +
            separator + "\n" +
            err + "\n" +
            separator + "\n" +
            "请根据上述错误信息修复您的 CUDA/C++ 代码。"
        )
    
    # 运行时/正确性错误 - 使用专门的运行时错误提取
    if not result.correctness:
        # 检查是否是输出不匹配（有详细信息）
        if md.get("correctness_issue") == "Output mismatch" and md.get("ref_output_shape"):
            # 构建详细的输出不匹配反馈
            separator = "=" * 50
            lines = [
                "您之前的回答编译成功，但输出结果与参考实现不匹配。",
                "",
                "输出对比详情:",
                separator,
                f"参考模型输出形状: {md.get('ref_output_shape')}",
                f"您的模型输出形状: {md.get('custom_output_shape')}",
                "",
                "数值差异:",
                f"  最大绝对差异: {md.get('max_difference', ['N/A'])[-1] if md.get('max_difference') else 'N/A'}",
                f"  平均绝对差异: {md.get('avg_difference', ['N/A'])[-1] if md.get('avg_difference') else 'N/A'}",
                f"  中位数绝对差异: {md.get('median_difference', ['N/A'])[-1] if md.get('median_difference') else 'N/A'}",
                f"  容差设置: atol={md.get('tolerance_atol', 'N/A')}, rtol={md.get('tolerance_rtol', 'N/A')}",
                "",
                "参考模型输出值范围:",
                f"  最小值: {md.get('ref_output_min', 'N/A')}",
                f"  最大值: {md.get('ref_output_max', 'N/A')}",
                "",
                "您的模型输出值范围:",
                f"  最小值: {md.get('custom_output_min', 'N/A')}",
                f"  最大值: {md.get('custom_output_max', 'N/A')}",
            ]
            
            # 添加样本差异
            sample_diffs = md.get("sample_differences", [])
            if sample_diffs:
                lines.extend([
                    "",
                    "差异最大的样本位置（前5个）:",
                ])
                for i, diff in enumerate(sample_diffs, 1):
                    lines.append(f"  {i}. 位置 {diff['index']}: 参考={diff['ref']}, 您的={diff['custom']}, 差异={diff['diff']}")
            
            lines.extend([
                "",
                separator,
                "可能的原因:",
                "- CUDA 内核中的数值计算错误（如错误的算术运算）",
                "- 线程索引计算错误导致数据访问错位",
                "- 边界条件处理不当（如 padding、stride 计算错误）",
                "- 内存访问模式错误（如 bank conflict 导致数据错乱）",
                "- 浮点数精度问题（如累加顺序不一致）",
                "",
                "请检查您的 CUDA 内核实现，确保:",
                "1. 线程索引映射正确对应输入/输出位置",
                "2. 边界条件（如边界检查）处理正确",
                "3. 数值计算与 PyTorch 参考实现一致",
            ])
            
            return "\n".join(lines)
        
        # 其他运行时/正确性错误
        err = md.get("runtime_error") or md.get("correctness_issue") or "未知运行时/正确性错误"
        # 提取运行时错误详情
        err = sanitize_error_message(err, error_type="runtime")
        separator = "=" * 50
        return (
            "您之前的回答编译成功，但存在运行时/正确性错误。"
            "以下是详细的错误信息：\n" +
            separator + "\n" +
            err + "\n" +
            separator + "\n" +
            "请检查您的 CUDA 内核实现，特别是内存访问和线程同步。"
        )
    
    # 正确情况 - 显示加速比
    speedup_str = "N/A"
    try:
        if result.runtime > 0 and result.ref_runtime > 0:
            speedup = float(result.ref_runtime) / float(result.runtime)
            speedup_str = f"{speedup:.4f}x"
    except Exception:
        pass
    
    return (
        "您之前的回答是正确的，但可以更快。"
        f"相对于基线您实现的加速比为：{speedup_str}"
    )


def build_feedback_block(history: list[AttemptRecord], include_code: bool = False) -> str:
    """
    从历史构建反馈块。
    
    参数：
        history: 之前的尝试列表
        include_code: 是否包含之前的代码（默认：False，按需求）
    """
    if not history:
        return ""
    
    blocks = []
    
    if include_code:
        blocks.append("以下是您之前的尝试：")
        for rec in history:
            blocks.append(f"\n```python\n{rec.kernel_code.strip()}\n```")
            if rec.summary:
                blocks.append(f"\n摘要：{rec.summary.strip()}")
            blocks.append(f"\n{format_feedback_from_eval_result(rec.eval_result).strip()}")
    else:
        # 仅包含反馈，不包含代码（按需求）
        blocks.append("以下是您之前尝试的反馈：")
        for i, rec in enumerate(history, 1):
            blocks.append(f"\n--- 尝试 {i} ---")
            if rec.summary:
                blocks.append(f"摘要：{rec.summary.strip()}")
            blocks.append(format_feedback_from_eval_result(rec.eval_result).strip())
    
    blocks.append("\n请修复问题并生成新的完整代码。")
    return "\n".join(blocks)


def build_restart_prompt(
    base_prompt: str,
    history: list[AttemptRecord],
    segment_summaries: list[str],
) -> str:
    """
    在长度限制后构建重新开始的提示（兼容旧版本）。
    
    包含基础任务、之前的尝试总结和错误。
    """
    lines = [
        base_prompt.rstrip(),
        "",
        "=" * 60,
        "注意：之前的对话达到了 token 限制。"
        "以下是您之前尝试的总结：",
        "",
    ]
    
    for i, (rec, summary) in enumerate(zip(history, segment_summaries), 1):
        lines.append(f"--- 之前尝试 {i} ---")
        if summary:
            lines.append(f"方法摘要：{summary}")
        lines.append(format_feedback_from_eval_result(rec.eval_result))
        lines.append("")
    
    lines.extend([
        "请重新启动您的推理过程并生成新的完整代码。",
        "",
    ])
    
    return "\n".join(lines)


def build_restart_prompt_for_length(
    base_prompt: str,
    last_code: str,
    last_error: str,
    history: list[AttemptRecord],
    segment_summaries: list[str],
) -> str:
    """
    当遇到长度限制时，构建包含完整信息的重新开始提示。
    
    包含：
    - 原始题目要求
    - 上一轮模型给出的代码（可能被截断）
    - 失败原因说明
    - 之前所有尝试的 summary
    
    参数：
        base_prompt: 原始题目提示
        last_code: 上一轮模型生成的代码（可能不完整）
        last_error: 失败原因描述
        history: 历史尝试记录
        segment_summaries: 每个段的摘要列表
    
    返回：
        构造的完整 restart prompt
    """
    lines = [
        "=" * 70,
        "【重要】对话重启通知",
        "=" * 70,
        "",
        "由于之前的对话达到 token 长度限制，我们需要重新开始对话。",
        "以下是对话重启前的重要信息汇总：",
        "",
        "-" * 70,
        "【1. 原始任务要求】",
        "-" * 70,
        base_prompt.rstrip(),
        "",
        "-" * 70,
        "【2. 上一轮尝试的代码】",
        "-" * 70,
        "注意：由于长度限制，上一轮代码可能被截断或不完整。",
        "```python",
        last_code if last_code else "<代码被截断或为空>",
        "```",
        "",
        "-" * 70,
        "【3. 失败原因】",
        "-" * 70,
        last_error,
        "",
    ]
    
    # 添加历史尝试的 summary（如果有）
    if history or segment_summaries:
        lines.extend([
            "-" * 70,
            "【4. 历史尝试总结】",
            "-" * 70,
            "",
        ])
        
        # 添加之前的 segment summaries
        for i, summary in enumerate(segment_summaries, 1):
            lines.append(f"【之前对话段 {i} 的摘要】")
            lines.append(summary[:500] if summary else "无摘要")
            lines.append("")
        
        # 添加当前段的历史尝试
        if history:
            lines.append("【当前对话段的尝试历史】")
            for i, rec in enumerate(history[-3:], 1):  # 只显示最近3次
                lines.append(f"  尝试 {i}:")
                if rec.summary:
                    lines.append(f"    摘要: {rec.summary[:200]}")
                lines.append(f"    结果: {'成功' if rec.eval_result.correctness else '失败'}")
                if not rec.eval_result.correctness and rec.eval_result.metadata:
                    error = rec.eval_result.metadata.get('correctness_issue') or \
                           rec.eval_result.metadata.get('runtime_error') or \
                           rec.eval_result.metadata.get('compilation_error')
                    if error:
                        lines.append(f"    错误: {str(error)[:100]}")
            lines.append("")
    
    lines.extend([
        "=" * 70,
        "【任务要求】",
        "=" * 70,
        "基于以上信息，请：",
        "1. 简化您的推理过程，避免冗长的中间步骤描述",
        "2. 生成简洁但完整的 CUDA 内核代码",
        "3. 确保代码可以直接编译运行",
        "4. 如果之前代码接近正确，请在此基础上精简修正",
        "",
        "请直接回复您的完整代码（包含 load_inline 调用）：",
        "```python",
        "# 您的代码 here",
        "```",
        "",
    ])
    
    return "\n".join(lines)


# =============================================================================
# 对话持久化
# =============================================================================

class ConversationState:
    """管理对话状态，包括长度限制重启的段和 reasoning content。
    
    使用字典存储所有分段的对话历史，key 为段索引，value 为对应对话消息列表。
    当前活动的段由 segment_index 指示。
    """
    
    def __init__(self, system_prompt: str = ""):
        self.segment_index: int = 0  # 当前段索引
        self.segment_summaries: list[str] = []  # 每段的摘要
        self.segments: dict[int, list[dict]] = {}  # 存储所有分段的对话历史 {segment_index: messages}
        self.turn_metrics: list[dict] = []  # 所有轮的指标（跨段）
        self.system_prompt: str = system_prompt
        
        # 初始化第 0 段
        self.segments[0] = []
        if system_prompt:
            self.segments[0].append({"role": "system", "content": system_prompt})
    
    @property
    def messages(self) -> list[dict]:
        """获取当前段的消息列表（向后兼容）。"""
        return self.segments.get(self.segment_index, [])
    
    def get_all_messages(self) -> list[dict]:
        """获取所有段的完整消息列表（按段顺序合并）。"""
        all_messages = []
        for seg_idx in sorted(self.segments.keys()):
            all_messages.extend(self.segments[seg_idx])
        return all_messages
    
    def get_segment_messages(self, segment_idx: int) -> list[dict]:
        """获取指定段的消息列表。"""
        return self.segments.get(segment_idx, [])
    
    def add_turn(
        self, 
        user_content: str, 
        assistant_content: str, 
        metric: dict,
        reasoning_content: str | None = None
    ):
        """
        向当前对话添加一轮。
        
        参数：
            user_content: 用户消息内容
            assistant_content: 助手消息内容（代码）
            metric: 评估指标
            reasoning_content: 推理模型的思考内容（可选）
        """
        current_messages = self.segments.setdefault(self.segment_index, [])
        current_messages.append({"role": "user", "content": user_content})
        assistant_msg = {"role": "assistant", "content": assistant_content}
        # 保存 reasoning_content 用于持久化，但不用于请求
        if reasoning_content:
            assistant_msg["reasoning_content"] = reasoning_content
        current_messages.append(assistant_msg)
        self.turn_metrics.append(metric)
    
    def update_last_metric(self, metric: dict):
        """
        更新最后一轮的评估指标。
        安全地检查列表长度，如果列表为空则添加指标。
        """
        if self.turn_metrics:
            self.turn_metrics[-1] = metric
        else:
            self.turn_metrics.append(metric)
    
    def validate_metrics(self) -> tuple[int, int]:
        """
        验证 turn_metrics 和对话轮数是否一致。
        
        返回：
            (metric_count, turn_count) - 用于调试的计数
        """
        # 计算总对话轮数（所有段的 user-assistant 对）
        total_turns = 0
        for seg_idx, messages in self.segments.items():
            # 排除 system 消息，只计算 user-assistant 对
            user_assistant_count = sum(1 for m in messages if m.get("role") in ("user", "assistant"))
            total_turns += user_assistant_count // 2  # 每轮包含 user + assistant
        
        return len(self.turn_metrics), total_turns
    
    def get_messages_for_request(self, include_reasoning: bool = False) -> list[dict]:
        """
        获取用于请求的消息列表。
        
        参数：
            include_reasoning: 是否包含 reasoning_content（默认 False 以节省 token）
        
        返回：
            用于 LLM 请求的消息列表
        """
        current_messages = self.segments.get(self.segment_index, [])
        if include_reasoning:
            return current_messages.copy()
        
        # 排除 reasoning_content 以节省 token
        filtered_messages = []
        for msg in current_messages:
            msg_copy = msg.copy()
            if "reasoning_content" in msg_copy:
                del msg_copy["reasoning_content"]
            filtered_messages.append(msg_copy)
        return filtered_messages
    
    def restart_for_length(self, summary: str, reasoning_summary: str | None = None):
        """
        达到长度限制后重新开始对话。
        创建新的对话段，保留之前段的完整历史。
        
        注意：此方法只重置对话状态，不添加新的 prompt。
        新的 prompt 应该通过 add_turn() 添加。
        
        参数：
            summary: 内容摘要
            reasoning_summary: 推理内容摘要（可选）
        """
        # 保存已完成段的摘要
        segment_summary = summary
        if reasoning_summary:
            segment_summary = f"{summary}\n\n推理过程：{reasoning_summary[:500]}"
        self.segment_summaries.append(segment_summary)
        
        # 增加段计数器，创建新段
        self.segment_index += 1
        self.segments[self.segment_index] = []
        
        # 新段只保留系统消息
        if self.system_prompt:
            self.segments[self.segment_index].append({"role": "system", "content": self.system_prompt})
    
    def to_dict(self) -> dict:
        """将状态序列化为字典。"""
        return {
            "turn_metrics": self.turn_metrics,
            "segment_index": self.segment_index,
            "segment_summaries": self.segment_summaries,
            "segments": self.segments,
            "system_prompt": self.system_prompt,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationState":
        """从字典恢复状态。"""
        # 使用保存的 system_prompt 初始化
        system_prompt = data.get("system_prompt", "")
        state = cls(system_prompt=system_prompt)
        state.turn_metrics = data.get("turn_metrics", [])
        state.segment_index = data.get("segment_index", 0)
        state.segment_summaries = data.get("segment_summaries", [])
        # 恢复 segments，将字符串 key 转换回 int key
        segments_data = data.get("segments", {})
        if isinstance(segments_data, dict):
            state.segments = {int(k): v for k, v in segments_data.items()}
        elif isinstance(segments_data, list):
            # 兼容旧格式（segments 是列表）
            state.segments = {i: v for i, v in enumerate(segments_data)}
        return state


def save_conversation(
    run_dir: str,
    level_label: str,
    problem_id: int,
    sample_id: int,
    state: ConversationState,
    verbose: bool = False,
) -> None:
    """将对话状态保存到 JSON。"""
    conv_path = os.path.join(
        run_dir,
        f"level_{level_label}_problem_{problem_id}_sample_{sample_id}_conversation.json",
    )
    
    # 验证 metrics 和对话轮数是否一致
    metric_count, turn_count = state.validate_metrics()
    if metric_count != turn_count and verbose:
        print(
            f"[Warning] Metrics ({metric_count}) 和对话轮数 ({turn_count}) 不匹配! "
            f"level={level_label} problem={problem_id} sample={sample_id}"
        )
    
    with open(conv_path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)


def load_conversation(
    run_dir: str,
    level_label: str,
    problem_id: int,
    sample_id: int,
) -> ConversationState | None:
    """从 JSON 加载对话状态（如果存在）。"""
    conv_path = os.path.join(
        run_dir,
        f"level_{level_label}_problem_{problem_id}_sample_{sample_id}_conversation.json",
    )
    if not os.path.exists(conv_path):
        return None
    
    try:
        with open(conv_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ConversationState.from_dict(data)
    except Exception:
        return None


# =============================================================================
# 核心生成逻辑
# =============================================================================

def generate_sample_multiturn_single(
    work: WorkArgs,
    config: GenerationMultiturnAllLevelsConfig,
    dataset,
    inference_server: Callable,
    run_dir: str,
) -> bool:
    """
    生成单个样本并进行多轮优化。
    
    支持：
    - 多轮反馈，不包含之前的代码
    - 达到长度限制时重新开始对话并保留摘要
    - 在 JSON 输出中跟踪段式对话
    """
    problem = dataset.get_problem_by_id(work.problem_id)
    ref_arch_src = problem.code
    problem_name = problem.name
    
    # 构建基础提示
    if config.custom_prompt_key:
        base_prompt = get_custom_prompt(
            config.custom_prompt_key,
            ref_arch_src=ref_arch_src,
            backend=config.backend,
            option=config.prompt_option,
            precision=config.precision,
            include_hardware=config.include_hardware_info,
            gpu_name=config.hardware_gpu_name,
        )
    else:
        base_prompt = get_prompt_for_backend(
            ref_arch_src,
            config.backend,
            option=config.prompt_option,
            precision=config.precision,
            include_hardware=config.include_hardware_info,
            gpu_name=config.hardware_gpu_name,
        )
    
    # 初始化或恢复对话状态
    state = load_conversation(run_dir, config.level_label, work.problem_id, work.sample_id)
    if state is None:
        state = ConversationState(system_prompt=config.conversation_system_prompt)
        start_turn = 0
        history: list[AttemptRecord] = []
    else:
        # 从保存状态恢复
        start_turn = len(state.turn_metrics)
        history = []
        for i, metric in enumerate(state.turn_metrics):
            # 尝试从轮次文件加载内核代码
            turn_path = os.path.join(
                run_dir,
                f"level_{config.level_label}_problem_{work.problem_id}_sample_{work.sample_id}_turn_{i}_kernel.py",
            )
            kernel_code = ""
            if os.path.exists(turn_path):
                with open(turn_path, "r", encoding="utf-8") as f:
                    kernel_code = f.read()
            
            # 从助手消息提取摘要（如果可用）
            # 注意：需要从所有段的消息中查找
            summary = None
            all_messages = state.get_all_messages()
            if i * 2 + 1 < len(all_messages):
                assistant_msg = all_messages[i * 2 + 1].get("content", "")
                summary = extract_summary_after_first_codeblock(assistant_msg)
            
            history.append(AttemptRecord(
                kernel_code=kernel_code,
                summary=summary,
                eval_result=turn_metric_to_eval_result(metric),
            ))
        
        if config.verbose:
            print(
                f"[MultiTurn] 恢复 level={config.level_label} problem={work.problem_id} "
                f"sample={work.sample_id} 从 turn {start_turn}, segment {state.segment_index}"
            )
    
    # 首轮后缀以鼓励代码块格式
    first_turn_suffix = (
        "\n\n请用 markdown 代码块回复您的完整代码"
        "（例如 ```python ... ```）。您可以在代码块后添加简要的变更摘要。"
    )
    
    last_kernel: Optional[str] = None
    
    # 使用独立的 turn 计数器，确保 turn 编号连续递增
    current_turn = start_turn
    
    for _ in range(start_turn, config.max_turns):
        turn = current_turn
        
        # 确定本轮的用户内容
        if turn == 0:
            user_content = base_prompt.rstrip() + first_turn_suffix
        elif state.segment_index > 0 and len(state.messages) == 2:
            # 重启后的第一轮 - 提示已在历史中
            user_content = None  # 将使用现有消息
        else:
            user_content = build_feedback_block(history, include_code=False)
        
        # 构建要发送的消息（排除 reasoning_content 以节省 token）
        base_messages = state.get_messages_for_request(include_reasoning=False)
        if user_content is not None:
            messages_to_send = base_messages + [{"role": "user", "content": user_content}]
        else:
            messages_to_send = base_messages
        
        # 如果需要则记录提示
        if config.log_prompt and user_content:
            prompt_path = os.path.join(
                run_dir,
                f"level_{config.level_label}_problem_{work.problem_id}_sample_{work.sample_id}_turn_{turn}_prompt.txt",
            )
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(user_content)
        
        # 查询 LLM
        response = inference_server(messages_to_send)
        raw_str = (response.content or "").strip()
        finish_reason = response.finish_reason
        reasoning_content = getattr(response, "reasoning_content", None)
        
        # 提取内核和摘要
        kernel = extract_first_code(raw_str, ["python", "cpp"])
        summary = extract_summary_after_first_codeblock(raw_str)
        
        # 处理长度限制或上下文窗口超限 - 重新开始对话
        needs_restart = finish_reason in ("length", "context_window_exceeded")
        
        if needs_restart:
            # 根据错误类型确定错误信息
            if finish_reason == "context_window_exceeded":
                error_msg = "请求超出上下文窗口限制。对话历史过长或 prompt 太大。"
                error_detail = getattr(response, 'error', error_msg)
            else:
                error_msg = "响应因达到 token 长度限制而被截断。模型生成的代码过长或推理过程过于详细。"
                error_detail = error_msg
            
            # 记录触发 restart 的这一轮（标记为生成失败）
            # 添加 generation_failed 字段以区分正常评估失败
            restart_eval_result = make_parsing_failure_result(
                f"{error_msg} 即将重启对话。"
            )
            restart_metric = eval_result_to_turn_metric(turn, state.segment_index, restart_eval_result)
            restart_metric["generation_failed"] = True  # 标记为生成阶段失败
            restart_metric["generation_error"] = error_msg
            state.add_turn(user_content or "", raw_str, restart_metric, reasoning_content)
            
            # 立即保存，确保触发 restart 的轮次被记录
            save_conversation(run_dir, config.level_label, work.problem_id, work.sample_id, state, config.verbose)
            
            if state.segment_index >= config.max_conversation_segments - 1:
                # 达到最大重启次数，视为本轮失败
                if config.verbose:
                    print(
                        f"[MultiTurn] 达到最大对话段数 "
                        f"level={config.level_label} problem={work.problem_id} sample={work.sample_id}"
                    )
                history.append(AttemptRecord(kernel_code=raw_str, summary=summary, eval_result=restart_eval_result))
                save_conversation(run_dir, config.level_label, work.problem_id, work.sample_id, state, config.verbose)
                continue
            
            # 获取上一轮模型给出的代码
            last_code = raw_str
            if not last_code and state.turn_metrics:
                current_messages = state.get_segment_messages(state.segment_index)
                for msg in reversed(current_messages):
                    if msg.get("role") == "assistant":
                        last_code = msg.get("content", "")
                        break
            
            # 构造 restart prompt
            restart_prompt_text = build_restart_prompt_for_length(
                base_prompt=base_prompt,
                last_code=last_code,
                last_error=error_detail,
                history=history,
                segment_summaries=state.segment_summaries,
            )
            
            # 重新开始对话
            state.restart_for_length(summary or "", reasoning_content)
            
            if config.verbose:
                restart_reason = "上下文窗口超限" if finish_reason == "context_window_exceeded" else "长度限制"
                print(
                    f"[MultiTurn] 因{restart_reason}重启对话 "
                    f"level={config.level_label} problem={work.problem_id} sample={work.sample_id} "
                    f"(segment {state.segment_index})"
                )
            
            # 构造消息：系统消息 + restart_prompt 作为 user 消息
            messages_to_send = state.get_messages_for_request(include_reasoning=False)
            messages_to_send.append({"role": "user", "content": restart_prompt_text})
            
            # 记录 restart 后的 prompt
            if config.log_prompt:
                restart_prompt_path = os.path.join(
                    run_dir,
                    f"level_{config.level_label}_problem_{work.problem_id}_sample_{work.sample_id}_turn_{turn}_restart_prompt.txt",
                )
                with open(restart_prompt_path, "w", encoding="utf-8") as f:
                    f.write(restart_prompt_text)
            
            # 重新查询 LLM
            response = inference_server(messages_to_send)
            raw_str = (response.content or "").strip()
            finish_reason = response.finish_reason
            reasoning_content = getattr(response, "reasoning_content", None)
            
            # 重新提取内核和摘要
            kernel = extract_first_code(raw_str, ["python", "cpp"])
            summary = extract_summary_after_first_codeblock(raw_str)
            
            # restart 消耗一个 turn 编号，为新尝试递增
            current_turn += 1
            # 更新 turn 变量以使用新的 turn 编号
            turn = current_turn
            
            # 如果 restart 后再次遇到 length/context_window_exceeded，记录警告
            if finish_reason in ("length", "context_window_exceeded"):
                if config.verbose:
                    print(
                        f"[MultiTurn] 警告：重启后仍然遇到 {finish_reason}，将尝试评估截断内容 "
                        f"level={config.level_label} problem={work.problem_id} sample={work.sample_id}"
                    )
            
            # 添加 restart 后的新 turn（使用新的 turn 编号，此时还没有评估结果）
            state.add_turn(restart_prompt_text, raw_str, {}, reasoning_content)
            
            # 立即保存 restart 后的对话状态
            save_conversation(run_dir, config.level_label, work.problem_id, work.sample_id, state, config.verbose)
            
        else:
            # 正常流程（非 length/context_window_exceeded 情况）- 将轮次添加到对话
            if user_content is not None:
                state.add_turn(user_content, raw_str, {}, reasoning_content)  # 指标将在评估后更新
            
            # 立即保存对话状态，确保每轮都被记录
            save_conversation(run_dir, config.level_label, work.problem_id, work.sample_id, state, config.verbose)
        
        # 处理其他 LLM 错误
        if finish_reason == "error":
            error_msg = getattr(response, 'error', 'LLM 调用失败')
            if config.verbose:
                print(
                    f"[MultiTurn] LLM 调用失败 "
                    f"level={config.level_label} problem={work.problem_id} sample={work.sample_id}: {error_msg}"
                )
            eval_result = make_parsing_failure_result(f"LLM 调用失败: {error_msg}")
            metric = eval_result_to_turn_metric(turn, state.segment_index, eval_result)
            state.update_last_metric(metric)
            history.append(AttemptRecord(kernel_code=raw_str or "<空>", summary=summary, eval_result=eval_result))
            save_conversation(run_dir, config.level_label, work.problem_id, work.sample_id, state, config.verbose)
            continue
        
        # 处理解析失败
        if not kernel:
            eval_result = make_parsing_failure_result("响应中未找到围栏代码块。")
            metric = eval_result_to_turn_metric(turn, state.segment_index, eval_result)
            state.update_last_metric(metric)  # 更新指标
            history.append(AttemptRecord(kernel_code=raw_str or "<空>", summary=summary, eval_result=eval_result))
            save_conversation(run_dir, config.level_label, work.problem_id, work.sample_id, state, config.verbose)
            continue
        
        # 保存轮次内核
        turn_kernel_path = os.path.join(
            run_dir,
            f"level_{config.level_label}_problem_{work.problem_id}_sample_{work.sample_id}_turn_{turn}_kernel.py",
        )
        with open(turn_kernel_path, "w", encoding="utf-8") as f:
            f.write(kernel)
        
        # 静态检查
        if config.check_kernel:
            ok, error, warnings = validate_kernel_static(kernel, backend=config.backend, precision=config.precision)
            if not ok:
                eval_result = make_parsing_failure_result(f"静态检查失败：{error}。警告：{warnings}")
                metric = eval_result_to_turn_metric(turn, state.segment_index, eval_result)
                state.update_last_metric(metric)
                history.append(AttemptRecord(kernel_code=kernel, summary=summary, eval_result=eval_result))
                save_conversation(run_dir, config.level_label, work.problem_id, work.sample_id, state, config.verbose)
                continue
        
        last_kernel = kernel
        
        # 评估内核
        try:
            eval_result = eval_with_retries(ref_arch_src, kernel, config=config)
        except Exception as e:
            if config.verbose:
                print(
                    f"[MultiTurn] 评估异常 level={config.level_label} "
                    f"problem={work.problem_id} sample={work.sample_id} turn={turn}: {e}"
                )
            eval_result = make_parsing_failure_result(f"评估引发异常：{type(e).__name__}: {e}")
        
        # 更新指标
        metric = eval_result_to_turn_metric(turn, state.segment_index, eval_result)
        state.update_last_metric(metric)
        
        if config.verbose:
            print(
                f"[MultiTurn] level={config.level_label} problem={work.problem_id} "
                f"sample={work.sample_id} turn={turn} segment={state.segment_index} "
                f"name={problem_name} compiled={eval_result.compiled} correct={eval_result.correctness}"
            )
        
        history.append(AttemptRecord(kernel_code=kernel, summary=summary, eval_result=eval_result))
        save_conversation(run_dir, config.level_label, work.problem_id, work.sample_id, state, config.verbose)
        
        # 每轮结束后清理内存
        cleanup_cuda_memory()
        
        # 成功时提前停止
        if config.early_stop_on_correct and eval_result.compiled and eval_result.correctness:
            break
        
        # 正常完成一轮，递增 turn 计数器
        current_turn += 1
    
    # 最终清理
    cleanup_cuda_memory()
    
    # 最终内核输出
    if last_kernel is None:
        # 保存对话状态后再抛出异常
        save_conversation(run_dir, config.level_label, work.problem_id, work.sample_id, state, config.verbose)
        raise RuntimeError(
            f"所有轮次都未能生成可解析的代码 problem {work.problem_id}: {problem_name}"
        )
    
    kernel_path = os.path.join(
        run_dir,
        f"level_{config.level_label}_problem_{work.problem_id}_sample_{work.sample_id}_kernel.py",
    )
    with open(kernel_path, "w", encoding="utf-8") as f:
        f.write(last_kernel)
    
    return True


def generate_sample_multiturn_launcher(
    work: WorkArgs,
    config: GenerationMultiturnAllLevelsConfig,
    dataset,
    inference_server: Callable,
    run_dir: str,
) -> bool | None:
    """带错误处理的启动器。"""
    try:
        return generate_sample_multiturn_single(work, config, dataset, inference_server, run_dir)
    except Exception as e:
        level_label = getattr(config, "level_label", "unknown")
        print(
            f"生成多轮样本错误 level={level_label} "
            f"problem={work.problem_id} sample={work.sample_id}: {e}"
        )
        return None


# =============================================================================
# 主入口
# =============================================================================

@pydra.main(base=GenerationMultiturnAllLevelsConfig)
def main(config: GenerationMultiturnAllLevelsConfig):
    """多轮生成所有级别的主入口。"""
    
    # 设置 CUDA 编译环境变量（在导入任何模型之前）
    max_jobs = int(getattr(config, "cuda_compile_max_jobs", 1))
    setup_compile_environment(max_jobs=max_jobs)
    if config.verbose:
        print(f"[Setup] CUDA 编译最大并行作业数设置为: {max_jobs}")
    
    # 解析级别设置
    level_raw = str(getattr(config, "level", "all")).strip().lower()
    single_level: Optional[int] = None
    if level_raw in ("1", "2", "3", "4"):
        single_level = int(level_raw)
    elif level_raw not in ("all", ""):
        raise ValueError("level 必须是 'all' 或 1/2/3/4 之一")
    
    # 解析布尔标志
    include_level4_expand = str(getattr(config, "include_level4_expand", False)).lower() in ("true", "1", "yes")
    config.is_reasoning_model = str(getattr(config, "is_reasoning_model", False)).lower() in ("true", "1", "yes")
    config.include_hardware_info = str(getattr(config, "include_hardware_info", False)).lower() in ("true", "1", "yes")
    
    # 应用服务器预设
    if config.server_type and config.server_type in SERVER_PRESETS:
        preset = SERVER_PRESETS[config.server_type]
        if config.model_name is None or config.model_name == "None":
            config.model_name = preset.get("model_name", "None")
        if config.max_tokens is None or config.max_tokens == "None":
            config.max_tokens = preset.get("max_tokens", "None")
        if config.temperature is None or config.temperature == "None":
            config.temperature = preset.get("temperature", "None")
    
    # 解析自定义提示键
    custom_prompt_key = getattr(config, "custom_prompt_key", None)
    if isinstance(custom_prompt_key, str):
        trimmed = custom_prompt_key.strip()
        custom_prompt_key = None if trimmed.lower() in ("", "none") else trimmed
    config.custom_prompt_key = custom_prompt_key
    
    # 验证后端
    supported_backends = {"cuda", "triton", "cute", "tilelang", "thunderkittens"}
    backend = config.backend.lower()
    if backend not in supported_backends:
        raise ValueError(f"不支持的后端：{config.backend}。必须是 {sorted(supported_backends)} 之一")
    config.backend = backend
    if backend == "tilelang":
        config.precision = "fp16"
    if backend == "thunderkittens":
        config.precision = "bf16"
    
    # 验证提示选项
    config.prompt_option = str(config.prompt_option).lower()
    valid_prompt_options = {"zero_shot", "one_shot", "few_shot"}
    if not config.custom_prompt_key and config.prompt_option not in valid_prompt_options:
        raise ValueError(f"无效的 prompt_option：{config.prompt_option}。必须是 {sorted(valid_prompt_options)} 之一")
    if config.include_hardware_info and not config.hardware_gpu_name:
        raise ValueError("include_hardware_info 为 True 但未提供 hardware_gpu_name")
    
    # 获取级别规格
    level_specs = get_all_level_specs(config.dataset_src, include_level4_expand)
    if single_level is not None:
        level_specs = [s for s in level_specs if s[0] == single_level]
        if not level_specs:
            raise ValueError(f"level={single_level} 不在 level_specs 中")
        print(f"单级别模式：仅级别 {single_level}")
    
    print(f"使用配置启动多轮批量生成：{config}")
    print(f"要生成的级别：{[s[2] for s in level_specs]}")
    
    # 创建推理服务器
    inference_server = create_inference_server_with_metadata(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        is_reasoning_model=config.is_reasoning_model,
        reasoning_effort=config.reasoning_effort,
        budget_tokens=config.budget_tokens,
    )
    
    # 设置运行目录
    run_dir = os.path.join(config.runs_dir, config.run_name)
    run_exists = os.path.exists(run_dir)
    
    if run_exists and run_dir_has_existing_kernels(run_dir):
        if not ask_resume(run_dir):
            print("要重新开始，请使用不同的 run_name。退出。")
            sys.exit(0)
        print("继续：将跳过已生成的内核。")
    elif run_exists:
        print(f"\n⚠️  运行目录已存在：{run_dir}")
        print("现有内核将被跳过。要重新开始请使用不同的 run_name。")
    
    os.makedirs(run_dir, exist_ok=True)
    
    # 保存配置
    save_dict = config.to_dict()
    save_dict["level"] = single_level if single_level is not None else "all"
    save_dict["include_level4_expand"] = include_level4_expand
    pydra.save_yaml(save_dict, os.path.join(run_dir, "generation_config.yaml"))
    
    assert config.store_type == "local", "仅支持本地存储"
    
    # 解析问题子集文件（如果提供）
    problem_subset_by_level = None
    if getattr(config, "problem_subset_file", None):
        path = config.problem_subset_file.strip()
        if os.path.isfile(path):
            problem_subset_by_level = parse_problem_subset_file(path)
            total_problems = sum(len(s) for s in problem_subset_by_level.values())
            print(f"使用问题子集文件：{path}（{total_problems} 个问题）")
        else:
            raise FileNotFoundError(f"problem_subset_file 未找到：{path}")
    
    # 生成统计
    total_generated = 0
    total_attempted = 0
    total_failed = 0
    total_skipped = 0
    
    # 处理每个级别
    for dataset_level, local_subdir, level_label in level_specs:
        config.level_label = level_label
        
        dataset = construct_kernelbench_dataset(
            level=dataset_level,
            source=config.dataset_src,
            dataset_name=config.dataset_name,
            local_subdir=local_subdir,
        )
        all_problem_ids = dataset.get_problem_ids()
        
        # 筛选问题
        if problem_subset_by_level is not None:
            allowed_ids = problem_subset_by_level.get(level_label)
            if not allowed_ids:
                print(f"级别 {level_label}：子集文件中无此级别，跳过。")
                continue
            problem_ids_to_run = [p for p in all_problem_ids if p in allowed_ids]
        elif config.subset == (None, None):
            problem_ids_to_run = all_problem_ids
        else:
            start, end = config.subset
            problem_ids_to_run = [p for p in all_problem_ids if start <= p <= end]
        
        if not problem_ids_to_run:
            print(f"警告：级别 {level_label} 没有要运行的问题，跳过。")
            continue
        
        # 构建工作列表
        problems_to_run: list[WorkArgs] = []
        level_total = 0
        level_skipped = 0
        
        for problem_id in problem_ids_to_run:
            for sample_id in range(config.num_samples):
                level_total += 1
                if check_kernel_exists(run_dir, level_label, problem_id, sample_id):
                    level_skipped += 1
                else:
                    problems_to_run.append(WorkArgs(problem_id=int(problem_id), sample_id=sample_id))
        
        if level_skipped > 0:
            print(f"📁 级别 {level_label}：{level_skipped}/{level_total} 内核已存在。")
        
        if not problems_to_run:
            print(f"级别 {level_label}：没有要生成的新内核。")
            total_skipped += level_total
            continue
        
        print(f"级别 {level_label}：生成 {len(problems_to_run)} 个内核（共 {level_total} 个）。")
        
        # 运行生成
        results = maybe_multithread(
            generate_sample_multiturn_launcher,
            problems_to_run,
            config.num_workers,
            time_interval=config.api_query_interval,
            config=config,
            dataset=dataset,
            inference_server=inference_server,
            run_dir=run_dir,
        )
        
        num_ok = sum(1 for r in results if r is True)
        num_fail = len(results) - num_ok
        
        total_generated += num_ok
        total_attempted += len(problems_to_run)
        total_failed += num_fail
        total_skipped += level_skipped
        
        print(f"级别 {level_label}：生成 {num_ok} 个，失败 {num_fail} 个，跳过 {level_skipped} 个。")
    
    # 最终摘要
    print("\n" + "=" * 60)
    print(
        f"所有级别：生成 {total_generated} 个，尝试 {total_attempted} 个，"
        f"失败 {total_failed} 个，跳过 {total_skipped} 个。"
    )
    
    if total_attempted == 0 and total_skipped > 0:
        print(f"✅ 所有内核已存在于 {run_dir}")
    elif total_failed > 0:
        print(f"请重试失败的 {total_failed} 个问题。")


if __name__ == "__main__":
    main()
