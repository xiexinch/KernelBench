"""
Batch evaluation for all levels (1, 2, 3, 4; optionally level4_expand).
Results are written to a single eval_results.json with per-level keys to avoid
problem_id collision across levels. Each script is independent; level list and
multi-level JSON logic are defined locally.
"""

import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from collections import defaultdict
from typing import Optional

import pydra
import torch
from pydra import Config, REQUIRED
from tqdm import tqdm

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import (
    check_metadata_serializable_all_types,
    eval_kernel_against_ref,
    get_error_name,
    KernelExecResult,
)
from kernelbench.utils import read_file, set_gpu_arch

from eval_from_generations_fix import (
    EvalConfig,
    WorkArgs,
    fetch_kernel_from_disk,
    fetch_ref_arch_from_problem_id,
    calculate_pass_at_k,
)

# kernelbench.eval for get_torch_dtype_from_string
from kernelbench import eval as eval_module

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_all_level_specs(
    dataset_src: str, include_level4_expand: bool
) -> list[tuple[int, Optional[str], str]]:
    """返回 (dataset_level, local_subdir, level_label) 列表。"""
    specs = [(1, None, "1"), (2, None, "2"), (3, None, "3"), (4, None, "4")]
    if include_level4_expand and dataset_src == "local":
        specs.append((4, "level4_expand", "level4_expand"))
    elif include_level4_expand and dataset_src != "local":
        raise ValueError("level4_expand is only supported with dataset_src=local")
    return specs


def level_label_to_level_key(level_label: str) -> str:
    """与 benchmark baseline 的 key 一致：level1, level2, ..., level4_expand。"""
    if level_label == "level4_expand":
        return "level4_expand"
    return f"level{level_label}"


def check_if_eval_exists_local(
    eval_file_path: str,
    level_key: str,
    problem_id: int,
    sample_id: int,
) -> bool:
    """多 level 格式：eval_results[level_key][str(problem_id)] 为 list，检查是否已有该 sample_id。"""
    if not os.path.exists(eval_file_path):
        return False
    try:
        with open(eval_file_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False
    # 多 level 格式：顶层 key 为 level1, level2 等
    if level_key not in data:
        return False
    level_data = data[level_key]
    key = str(problem_id)
    if key not in level_data:
        return False
    entry = level_data[key]
    if isinstance(entry, list):
        return any(r.get("sample_id") == sample_id for r in entry)
    return entry.get("sample_id") == sample_id


def add_to_eval_results_file(
    problem_id: int,
    sample_id: int,
    eval_result: KernelExecResult,
    eval_file_path: str,
    level_key: str,
):
    """多 level 格式：写入 eval_results[level_key][str(problem_id)] 的 list。"""
    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r") as f:
            eval_results = json.load(f)
    else:
        eval_results = {}

    if level_key not in eval_results:
        eval_results[level_key] = {}
    level_data = eval_results[level_key]
    key = str(problem_id)
    if key not in level_data:
        level_data[key] = []
    level_data[key].append(
        {
            "sample_id": sample_id,
            "compiled": eval_result.compiled,
            "correctness": eval_result.correctness,
            "metadata": check_metadata_serializable_all_types(eval_result.metadata),
            "runtime": eval_result.runtime,
            "runtime_stats": eval_result.runtime_stats,
        }
    )
    # 每个 level 内按 problem_id 数字排序
    eval_results[level_key] = dict(
        sorted(level_data.items(), key=lambda x: int(x[0]))
    )
    os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
    with open(eval_file_path, "w") as f:
        json.dump(eval_results, f, indent=4)


def remove_cache_dir(
    cache_dir: str,
    run_name: str,
    level_key: str,
    problem_id: int,
    sample_id: int,
):
    """删除带 level_key 的 cache 目录。"""
    problem_cache_dir = os.path.join(
        cache_dir, run_name, level_key, str(problem_id), str(sample_id)
    )
    if os.path.exists(problem_cache_dir):
        try:
            shutil.rmtree(problem_cache_dir, ignore_errors=True)
            print(
                f"\n[INFO] Removed cached folder for level {level_key} problem {problem_id} sample {sample_id}"
            )
        except Exception as e:
            print(
                f"\n[WARNING] Failed to remove cache directory {problem_cache_dir}: {str(e)}"
            )


def evaluate_single_sample(
    work_args: WorkArgs,
    config: EvalConfig,
    dataset,
    run_dir: str,
    level_key: str,
) -> KernelExecResult | None:
    """单样本评测，build_dir 含 level_key 避免跨 level 冲突。"""
    problem_id, sample_id, device = (
        work_args.problem_id,
        work_args.sample_id,
        work_args.device,
    )
    ref_arch_src = fetch_ref_arch_from_problem_id(
        dataset, problem_id, config.dataset_src
    )
    kernel_src = fetch_kernel_from_disk(
        run_dir, config.level_label, problem_id, sample_id
    )
    if kernel_src is None:
        print(
            f"[WARNING] Kernel not found for problem {problem_id} sample {sample_id}, skipping"
        )
        return KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={"error": "Kernel file not found"},
            runtime=-1.0,
            runtime_stats={},
        )
    build_dir = os.path.join(
        config.kernel_eval_build_dir,
        config.run_name,
        level_key,
        str(problem_id),
        str(sample_id),
    )
    try:
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=config.measure_performance,
            timing_method=config.timing_method,
            verbose=config.verbose,
            num_correct_trials=config.num_correct_trials,
            num_perf_trials=config.num_perf_trials,
            build_dir=build_dir,
            device=device,
            backend=config.backend,
            precision=eval_module.get_torch_dtype_from_string(config.precision),
        )
        return eval_result
    except Exception as e:
        print(
            f"[WARNING] Evaluation failed for problem {problem_id} sample {sample_id}: {e}"
        )
        if "CUDA error" in str(e):
            metadata = {
                "cuda_error": f"CUDA Error: {str(e)}",
                "cuda_error_name": get_error_name(e),
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }
        else:
            metadata = {
                "other_error": f"error: {str(e)}",
                "other_error_name": get_error_name(e),
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }
        return KernelExecResult(
            compiled=False,
            correctness=False,
            metadata=metadata,
            runtime=-1.0,
            runtime_stats={},
        )


def batch_eval_all_levels(
    total_work: list[tuple[int, int]],
    config: EvalConfig,
    curr_level_dataset,
    run_dir: str,
    eval_file_path: str,
    level_key: str,
):
    """本地 GPU 批评测，结果按 level_key 写入同一 eval_results.json。"""
    batch_size = config.num_gpu_devices
    with tqdm(total=len(total_work), desc=f"Eval level {level_key}") as pbar:
        while len(total_work) > 0:
            curr_work_batch = total_work[:batch_size]
            total_work = total_work[batch_size:]
            print(
                f"[Curr Batch] {len(curr_work_batch)} tasks; [Total Work left] {len(total_work)}"
            )
            assert len(curr_work_batch) <= batch_size
            with mp.Pool(batch_size) as pool:
                work_args = [
                    (
                        WorkArgs(
                            problem_id=p_id,
                            sample_id=s_idx,
                            device=torch.device(f"cuda:{i % batch_size}"),
                        ),
                        config,
                        curr_level_dataset,
                        run_dir,
                        level_key,
                    )
                    for i, (p_id, s_idx) in enumerate(curr_work_batch)
                ]
                start_time = time.time()
                async_results = [
                    pool.apply_async(evaluate_single_sample, w) for w in work_args
                ]
                results = []
                batch_timeout = config.timeout
                for i, async_result in enumerate(async_results):
                    problem_id, sample_id = curr_work_batch[i]
                    try:
                        elapsed = time.time() - start_time
                        remaining = max(0, batch_timeout - elapsed)
                        result = async_result.get(timeout=remaining)
                        if result is None:
                            result = KernelExecResult(
                                compiled=False,
                                correctness=False,
                                metadata={"error": "Evaluation returned None"},
                                runtime=-1.0,
                                runtime_stats={},
                            )
                        results.append((problem_id, sample_id, result))
                    except (mp.TimeoutError, TimeoutError):
                        print(
                            f"[WARNING] Evaluation TIMED OUT for problem {problem_id} sample {sample_id}"
                        )
                        timeout_entry = (
                            problem_id,
                            sample_id,
                            KernelExecResult(
                                compiled=False,
                                correctness=False,
                                metadata={"error": "Evaluation timed out"},
                                runtime=-1.0,
                                runtime_stats={},
                            ),
                        )
                        results.append(timeout_entry)
                        remove_cache_dir(
                            config.kernel_eval_build_dir,
                            config.run_name,
                            level_key,
                            problem_id,
                            sample_id,
                        )
                    except Exception as e:
                        print(
                            f"[ERROR] Evaluation FAILED for problem {problem_id} sample {sample_id}: {e}"
                        )
                        fail_entry = (
                            problem_id,
                            sample_id,
                            KernelExecResult(
                                compiled=False,
                                correctness=False,
                                metadata={"error": str(e)},
                                runtime=-1.0,
                                runtime_stats={},
                            ),
                        )
                        results.append(fail_entry)
                        remove_cache_dir(
                            config.kernel_eval_build_dir,
                            config.run_name,
                            level_key,
                            problem_id,
                            sample_id,
                        )
                for problem_id, sample_id, result in results:
                    print(
                        f"[Eval Result] Level {level_key} problem {problem_id} sample {sample_id}: {result}"
                    )
                    try:
                        add_to_eval_results_file(
                            problem_id,
                            sample_id,
                            result,
                            eval_file_path,
                            level_key,
                        )
                    except Exception as save_err:
                        print(
                            f"[ERROR] Failed to save result: {save_err}"
                        )
                print(
                    f"[Curr batch] Took {time.time() - start_time:.2f} seconds"
                )
                pbar.update(len(curr_work_batch))


class EvalAllLevelsConfig(EvalConfig):
    """Config for evaluating all levels; level defaults to 'all'."""

    def __init__(self):
        super().__init__()
        self.level = "all"
        self.include_level4_expand = False


@pydra.main(base=EvalAllLevelsConfig)
def main(config: EvalAllLevelsConfig):
    """一次评测所有 level，结果写入同一 eval_results.json（按 level 分桶，题目编号不冲突）。"""
    print(f"Starting Batch Eval (all levels) with config: {config}")

    backend = config.backend.lower()
    if backend == "thunderkittens":
        config.precision = "bf16"
        config.gpu = "H100"
        print("[ThunderKittens] Auto-configured: precision=bf16, gpu=H100")

    if config.eval_mode == "local":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device not available. Local evaluation requires GPU."
            )
        set_gpu_arch(config.gpu_arch)
        assert config.num_gpu_devices <= torch.cuda.device_count(), (
            f"Requested {config.num_gpu_devices} GPUs, only {torch.cuda.device_count()} available."
        )
    else:
        print(f"[Modal] Using Modal with GPU: {config.gpu}")

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    level_raw = getattr(config, "level", "all")
    if str(level_raw).strip().lower() not in ("all", ""):
        raise ValueError(
            "本脚本仅支持评测所有 level。单 level 请使用 eval_from_generations_fix.py 并指定 level。"
        )
    include_level4_expand = getattr(config, "include_level4_expand", False)
    if isinstance(include_level4_expand, str):
        include_level4_expand = include_level4_expand.lower() in (
            "true",
            "1",
            "yes",
        )

    level_specs = get_all_level_specs(config.dataset_src, include_level4_expand)
    run_dir = os.path.join(config.runs_dir, config.run_name)
    eval_file_path = os.path.join(run_dir, "eval_results.json")

    for dataset_level, local_subdir, level_label in level_specs:
        config.level_label = level_label
        level_key = level_label_to_level_key(level_label)
        dataset = construct_kernelbench_dataset(
            level=dataset_level,
            source=config.dataset_src,
            dataset_name=config.dataset_name,
            local_subdir=local_subdir,
        )
        all_problem_ids = dataset.get_problem_ids()
        if config.problem_ids is not None:
            problem_ids_to_run = [
                p for p in config.problem_ids if p in all_problem_ids
            ]
        elif config.subset == (None, None):
            problem_ids_to_run = all_problem_ids
        else:
            start, end = config.subset
            problem_ids_to_run = [
                p for p in all_problem_ids if start <= p <= end
            ]
        if not problem_ids_to_run:
            print(
                f"Warning: No problems in subset for level {level_label}, skipping."
            )
            continue

        total_work = []
        total_tasks = 0
        already_evaluated = 0
        kernel_missing = 0
        for problem_id in problem_ids_to_run:
            for sample_id in range(config.num_samples_per_problem):
                total_tasks += 1
                if check_if_eval_exists_local(
                    eval_file_path, level_key, problem_id, sample_id
                ):
                    already_evaluated += 1
                    continue
                if fetch_kernel_from_disk(
                    run_dir, level_label, problem_id, sample_id
                ) is None:
                    kernel_missing += 1
                    continue
                total_work.append((problem_id, sample_id))

        if already_evaluated > 0:
            print(
                f"📁 Level {level_label}: {already_evaluated}/{total_tasks} already evaluated."
            )
        if kernel_missing > 0:
            print(
                f"⚠️  Level {level_label}: skipped {kernel_missing} (kernel not found)."
            )
        if not total_work:
            print(f"Level {level_label}: no new samples to evaluate.")
            continue

        print(
            f"Level {level_label}: evaluating {len(total_work)} samples (of {total_tasks})."
        )
        if config.eval_mode == "modal":
            raise NotImplementedError(
                "eval_from_generations_all_levels.py 暂不支持 eval_mode=modal，请使用 eval_mode=local 或对单 level 使用 eval_from_generations_fix.py 的 modal。"
            )
        batch_eval_all_levels(
            total_work,
            config,
            dataset,
            run_dir,
            eval_file_path,
            level_key,
        )

    # Pass@k: 按 level 分别计算后合并写入 pass_at_k_results.json
    if not os.path.exists(eval_file_path):
        print("No eval results file, skipping pass@k.")
        return
    with open(eval_file_path, "r") as f:
        all_eval_results = json.load(f)
    pass_at_k_merged = {}
    for level_key in all_eval_results:
        level_data = all_eval_results[level_key]
        if not level_data:
            continue
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(level_data, tmp, indent=2)
            tmp_path = tmp.name
        try:
            result = calculate_pass_at_k(
                tmp_path, config.pass_at_k_values
            )
            pass_at_k_merged[level_key] = result
        finally:
            os.unlink(tmp_path)
    pass_at_k_path = os.path.join(run_dir, "pass_at_k_results.json")
    with open(pass_at_k_path, "w") as f:
        json.dump(pass_at_k_merged, f, indent=2)
    print(f"Pass@k (per level) written to {pass_at_k_path}")


if __name__ == "__main__":
    main()
