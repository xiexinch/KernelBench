"""
generate_baseline_time_resume.py - Baseline time generation with resume support

Based on generate_baseline_time.py, adds:
- Resume: load existing results, skip problems that already have valid results
- Only measure failed (null) or missing problems
- Merge new results with existing and save
"""

import torch
import numpy as np
from kernelbench.dataset import (
    construct_kernelbench_dataset,
    fetch_ref_arch_from_dataset,
)
from kernelbench.timing import measure_ref_program_time
from kernelbench.utils import read_file
import os
import json
from tqdm import tqdm

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

TIMING_DIR = os.path.join(REPO_TOP_PATH, "results", "timing")


def _is_valid_baseline_result(result) -> bool:
    """Check if a baseline result is valid (non-null and has mean)."""
    if result is None:
        return False
    if not isinstance(result, dict):
        return False
    return "mean" in result and result["mean"] is not None


def _load_existing_results(save_path: str) -> dict:
    """Load existing results if file exists, else return empty structure."""
    if not os.path.exists(save_path):
        return {}
    try:
        with open(save_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def record_baseline_times_resume(
    use_torch_compile: bool = False,
    torch_compile_backend: str = "inductor",
    torch_compile_options: str = "default",
    file_name: str = "baseline_time.json",
    precision: str = "fp32",
):
    """
    Generate baseline time for KernelBench with resume support.
    Loads existing results, skips problems with valid results, only measures failed/missing.
    """
    device = torch.device("cuda:0")
    save_path = os.path.join(TIMING_DIR, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load existing results
    json_results = _load_existing_results(save_path)
    if not json_results:
        json_results = {}

    for level in [1, 2, 3]:
        level_key = f"level{level}"
        if level_key not in json_results:
            json_results[level_key] = {}

        dataset = construct_kernelbench_dataset(level)
        problem_ids = dataset.get_problem_ids()

        total = len(problem_ids)
        to_measure = []
        for problem_id in problem_ids:
            ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(
                dataset, problem_id
            )
            existing = json_results[level_key].get(ref_arch_name)
            if not _is_valid_baseline_result(existing):
                to_measure.append((problem_id, ref_arch_path, ref_arch_name, ref_arch_src))

        if not to_measure:
            print(f"[{level_key}] All {total} problems already have valid results, skipping.")
            continue

        print(f"[{level_key}] {len(to_measure)}/{total} problems need measurement (resuming)")
        for problem_id, ref_arch_path, ref_arch_name, ref_arch_src in tqdm(
            to_measure, desc=f"Level {level}"
        ):
            runtime_stats = measure_ref_program_time(
                ref_arch_name=ref_arch_name,
                ref_arch_src=ref_arch_src,
                use_torch_compile=use_torch_compile,
                torch_compile_backend=torch_compile_backend,
                torch_compile_options=torch_compile_options,
                device=device,
                verbose=False,
                precision=precision,
            )
            json_results[level_key][ref_arch_name] = runtime_stats

            # Save after each problem to avoid losing progress on crash
            with open(save_path, "w") as f:
                json.dump(json_results, f, indent=4)

    return json_results


def test_measure_particular_program(level_num: int, problem_id: int):
    """
    Test measure_program_time on a particular program
    """
    device = torch.device("cuda:0")

    dataset = construct_kernelbench_dataset(level_num)

    ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(
        dataset, problem_id
    )

    exec_stats = measure_ref_program_time(
        ref_arch_name=ref_arch_name,
        ref_arch_src=ref_arch_src,
        use_torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_options="default",
        device=device,
        verbose=False,
        precision="bf16",
    )

    print(f"Execution time for {ref_arch_name}: {exec_stats}")


if __name__ == "__main__":
    # Replace this with whatever hardware you are running on
    # hardware_name = "L40S_matx3"
    # hardware_name = "H100_PCIe_LambdaLabs"
    hardware_name = "H200"

    input(
        f"You are about to start recording baseline time for {hardware_name} (with resume). "
        f"Press Enter to continue..."
    )

    save_dir = os.path.join(TIMING_DIR, hardware_name)
    if os.path.exists(save_dir):
        print(f"📁 Found existing results in {save_dir}. Will resume - only measure failed/missing problems.")

    # 1. Record Torch Eager
    record_baseline_times_resume(
        use_torch_compile=False,
        torch_compile_backend=None,
        torch_compile_options=None,
        file_name=f"{hardware_name}/baseline_time_torch.json",
        precision="bf16",
    )

    # 2. Record Torch Compile using Inductor
    for torch_compile_mode in [
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ]:
        record_baseline_times_resume(
            use_torch_compile=True,
            torch_compile_backend="inductor",
            torch_compile_options=torch_compile_mode,
            file_name=f"{hardware_name}/baseline_time_torch_compile_inductor_{torch_compile_mode}.json",
            precision="bf16",
        )

    # 3. Record Torch Compile using cudagraphs
    record_baseline_times_resume(
        use_torch_compile=True,
        torch_compile_backend="cudagraphs",
        torch_compile_options=None,
        file_name=f"{hardware_name}/baseline_time_torch_compile_cudagraphs.json",
        precision="bf16",
    )

    print(f"\n✅ Baseline time saved to {save_dir}")
