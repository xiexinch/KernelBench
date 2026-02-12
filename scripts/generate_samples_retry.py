"""
仅对“参考 run”中失败或缺失的任务重新采样生成 kernel。

用法：
  通过 retry_from_run 指定之前跑过的任务（即 runs/${retry_from_run} 目录名）。
  脚本会读取 runs/${retry_from_run}/eval_results.json 与 generation_config.yaml，
  只对“未通过（compiled=false 或 correctness=false）或未出现在 eval 中”的 (problem_id, sample_id) 生成。

  多次采样：可指定 run_name 为新的目录（如 my-run-retry2），或使用默认的 ${retry_from_run}_retry；
  已存在的 kernel 文件会被跳过，因此可多次执行以补全失败任务。

示例：
  # 仅重跑 deepseek-chat 中失败/缺失的任务，结果写到 runs/deepseek-chat_retry
  python scripts/generate_samples_retry.py retry_from_run=deepseek-chat dataset_src=huggingface level=1 server_type=deepseek

  # 指定新 run 目录并重跑
  python scripts/generate_samples_retry.py retry_from_run=deepseek-chat run_name=deepseek-chat-retry2 dataset_src=huggingface level=1 server_type=deepseek
"""

import json
import os
import sys
from dataclasses import dataclass

import pydra
import torch
import yaml
from pydra import Config, REQUIRED

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend, get_custom_prompt
from kernelbench.utils import (
    create_inference_server_from_presets,
    extract_first_code,
    maybe_multithread,
)
from kernelbench.kernel_static_checker import validate_kernel_static

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import generate_samples as _gen

# 复用原有脚本的生成逻辑与配置结构（同目录 generate_samples）
GenerationConfig = _gen.GenerationConfig
WorkArgs = _gen.WorkArgs
check_kernel_exists = _gen.check_kernel_exists
generate_sample_launcher = _gen.generate_sample_launcher

torch.set_printoptions(precision=4, threshold=10)


def load_eval_results(runs_dir: str, run_name: str) -> dict:
    """加载 runs/${run_name}/eval_results.json，不存在或解析失败返回空 dict。"""
    path = os.path.join(runs_dir, run_name, "eval_results.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_ref_generation_config(runs_dir: str, run_name: str) -> dict | None:
    """加载 runs/${run_name}/generation_config.yaml。"""
    path = os.path.join(runs_dir, run_name, "generation_config.yaml")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def resolve_level_from_config(level_raw) -> tuple[int, str | None, str]:
    """与 generate_samples 一致：根据 level 解析出 dataset_level, local_subdir, level_label。"""
    if isinstance(level_raw, str) and level_raw and level_raw.strip().lower() == "level4_expand":
        return 4, "level4_expand", "level4_expand"
    return int(level_raw), None, str(int(level_raw))


def get_failed_or_missing(
    ref_config: dict,
    eval_results: dict,
    dataset_src: str,
    dataset_name: str,
) -> set[tuple[int, int]]:
    """
    根据参考 run 的 generation_config 与 eval_results，得到应重跑的 (problem_id, sample_id) 集合。
    - 缺失：该 (problem_id, sample_id) 在 eval_results 中无记录。
    - 失败：该条目的 compiled 为 false 或 correctness 为 false。
    """
    level_raw = ref_config.get("level")
    if level_raw is None:
        return set()
    dataset_level, local_subdir, level_label = resolve_level_from_config(level_raw)
    subset = ref_config.get("subset") or (None, None)
    if isinstance(subset, list):
        subset = tuple(subset) if len(subset) >= 2 else (None, None)
    num_samples = int(ref_config.get("num_samples") or 1)

    dataset = construct_kernelbench_dataset(
        level=dataset_level,
        source=dataset_src,
        dataset_name=dataset_name,
        local_subdir=local_subdir,
    )
    all_problem_ids = dataset.get_problem_ids()
    if subset == (None, None) or (subset[0] is None and subset[1] is None):
        problem_ids = all_problem_ids
    else:
        start, end = subset[0], subset[1]
        problem_ids = [p for p in all_problem_ids if start <= p <= end]

    to_retry = set()
    for problem_id in problem_ids:
        key = str(problem_id)
        samples_in_eval = {s["sample_id"] for s in eval_results.get(key, [])} if isinstance(eval_results.get(key), list) else set()
        for sample_id in range(num_samples):
            if sample_id not in samples_in_eval:
                to_retry.add((problem_id, sample_id))
                continue
            entries = [e for e in eval_results[key] if e.get("sample_id") == sample_id]
            if not entries:
                to_retry.add((problem_id, sample_id))
                continue
            e = entries[0]
            if not e.get("compiled", True) or not e.get("correctness", True):
                to_retry.add((problem_id, sample_id))
    return to_retry


class RetryGenerationConfig(Config):
    """仅重跑失败/缺失任务的生成配置。"""

    def __init__(self):
        self.retry_from_run = REQUIRED  # 参考 run 的 run_name，即 runs/${retry_from_run}
        self.run_name = None  # 可选，默认为 retry_from_run + "_retry"
        self.dataset_src = REQUIRED
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = None  # 可选，不填则从参考 run 的 generation_config 读取
        self.subset = (None, None)
        self.num_samples = None  # 可选，不填则从参考 run 读取
        self.num_workers = 64
        self.api_query_interval = 0.0
        self.server_type = None
        self.model_name = None
        self.max_tokens = None
        self.temperature = 0.0
        self.is_reasoning_model = False
        self.reasoning_effort = "low"
        self.budget_tokens = 0
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")
        self.verbose = False
        self.store_type = "local"
        self.log_prompt = False
        self.backend = "cuda"
        self.precision = "fp32"
        self.prompt_option = "one_shot"
        self.include_hardware_info = False
        self.hardware_gpu_name = None
        self.custom_prompt_key = None
        self.check_kernel = True

    def __repr__(self):
        return f"RetryGenerationConfig({self.to_dict()})"


@pydra.main(base=RetryGenerationConfig)
def main(config: RetryGenerationConfig):
    from kernelbench.utils import SERVER_PRESETS

    if config.server_type and config.server_type in SERVER_PRESETS:
        preset = SERVER_PRESETS[config.server_type]
        if config.model_name is None or config.model_name == "None":
            config.model_name = preset.get("model_name", "None")
        if config.max_tokens is None or config.max_tokens == "None":
            config.max_tokens = preset.get("max_tokens", "None")
        if config.temperature is None or config.temperature == "None":
            config.temperature = preset.get("temperature", "None")

    if isinstance(config.is_reasoning_model, str):
        config.is_reasoning_model = config.is_reasoning_model.lower() in ["true", "1", "yes"]
    custom_prompt_key = getattr(config, "custom_prompt_key", None)
    if isinstance(custom_prompt_key, str):
        trimmed = custom_prompt_key.strip()
        config.custom_prompt_key = None if trimmed.lower() in {"", "none"} else trimmed
    include_hardware = config.include_hardware_info
    if isinstance(include_hardware, str):
        include_hardware = include_hardware.lower() in ["true", "1", "yes"]
    config.include_hardware_info = include_hardware

    backend = (config.backend or "cuda").lower()
    if backend == "tilelang":
        config.precision = "fp16"
    if backend == "thunderkittens":
        config.precision = "bf16"
    config.prompt_option = str(config.prompt_option or "one_shot").lower()

    eval_results = load_eval_results(config.runs_dir, config.retry_from_run)
    ref_config = load_ref_generation_config(config.runs_dir, config.retry_from_run)
    if not ref_config:
        raise FileNotFoundError(
            f"未找到参考 run 配置: {os.path.join(config.runs_dir, config.retry_from_run, 'generation_config.yaml')}"
        )

    if config.run_name is None or (isinstance(config.run_name, str) and config.run_name.strip() == ""):
        config.run_name = f"{config.retry_from_run}_retry"
    if isinstance(config.run_name, str):
        config.run_name = config.run_name.strip()

    level_raw = config.level if config.level is not None else ref_config.get("level")
    if level_raw is None:
        raise ValueError("level 未在参考 run 的 generation_config 中找到，请显式传入 level=...")
    dataset_level, local_subdir, level_label = resolve_level_from_config(level_raw)
    if config.dataset_src != "local" and level_label == "level4_expand":
        raise ValueError("level4_expand 仅支持 dataset_src=local")

    num_samples = config.num_samples if config.num_samples is not None else int(ref_config.get("num_samples") or 1)
    config.num_samples = num_samples

    to_retry = get_failed_or_missing(
        ref_config, eval_results, config.dataset_src, config.dataset_name
    )
    if not to_retry:
        print(f"参考 run {config.retry_from_run} 中无失败或缺失任务，无需重跑。")
        return

    print(f"参考 run: {config.retry_from_run}，共 {len(to_retry)} 个 (problem_id, sample_id) 需重跑。")

    dataset = construct_kernelbench_dataset(
        level=dataset_level,
        source=config.dataset_src,
        dataset_name=config.dataset_name,
        local_subdir=local_subdir,
    )

    run_dir = os.path.join(config.runs_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    gen_config = GenerationConfig()
    for k, v in config.to_dict().items():
        if hasattr(gen_config, k):
            setattr(gen_config, k, v)
    gen_config.level = level_raw
    gen_config.level_label = level_label
    gen_config.subset = config.subset if config.subset != (None, None) else (None, None)
    gen_config.num_samples = num_samples
    gen_config.run_name = config.run_name
    # 避免 custom_prompt_key 从 to_dict 得到字符串 "None"，导致 get_custom_prompt 报错
    cpk = getattr(gen_config, "custom_prompt_key", None)
    if isinstance(cpk, str) and cpk.strip().lower() in {"", "none"}:
        gen_config.custom_prompt_key = None

    problems_to_run = []
    for (problem_id, sample_id) in sorted(to_retry):
        if not check_kernel_exists(run_dir, level_label, problem_id, sample_id):
            problems_to_run.append(WorkArgs(problem_id=problem_id, sample_id=sample_id))

    pydra.save_yaml(gen_config.to_dict(), os.path.join(run_dir, "generation_config.yaml"))

    if not problems_to_run:
        print(f"所有需重跑任务在 {run_dir} 中已存在 kernel，无需生成。")
        return

    print(f"本次将生成 {len(problems_to_run)} 个 kernel 到 {run_dir}。")
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        is_reasoning_model=config.is_reasoning_model,
        reasoning_effort=config.reasoning_effort,
        budget_tokens=config.budget_tokens,
    )

    generation_results = maybe_multithread(
        generate_sample_launcher,
        problems_to_run,
        config.num_workers,
        time_interval=config.api_query_interval,
        config=gen_config,
        dataset=dataset,
        inference_server=inference_server,
        run_dir=run_dir,
    )

    num_ok = len(generation_results)
    num_failed = len(problems_to_run) - num_ok
    print(f"\n已生成 {num_ok}/{len(problems_to_run)} 个 kernel。失败 {num_failed} 个，可再次执行本脚本重试。")


if __name__ == "__main__":
    main()
