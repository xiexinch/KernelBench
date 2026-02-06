"""
generate_baseline_time_resume.py - Baseline time generation with resume support

Based on generate_baseline_time.py, adds:
- Resume: load existing results, skip problems that already have valid results
- Only measure failed (null) or missing problems
- Merge new results with existing and save
- num_gpus: parallel measurement across multiple GPUs
- timeout: per-operator timeout, terminate hung process
"""

import torch
import numpy as np
import multiprocessing as mp
import threading
from queue import Queue, Empty
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


def _measure_one_worker(args) -> tuple:
    """
    Worker for multiprocessing: measure one problem on a given GPU.
    args: (ref_arch_name, ref_arch_src, device_id, use_torch_compile, torch_compile_backend, torch_compile_options, precision)
    Returns: (ref_arch_name, runtime_stats or None)
    """
    (
        ref_arch_name,
        ref_arch_src,
        device_id,
        use_torch_compile,
        torch_compile_backend,
        torch_compile_options,
        precision,
    ) = args
    device = torch.device(f"cuda:{device_id}")
    try:
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
        return (ref_arch_name, runtime_stats)
    except Exception as e:
        print(f"[WARNING] measure_ref_program_time failed for {ref_arch_name}: {e}")
        return (ref_arch_name, None)


def _measure_worker_with_queue(work_item: tuple, result_queue: mp.Queue) -> None:
    """
    Module-level worker for Process: run _measure_one_worker and put result in queue.
    Must be at module level for picklability (spawn context).
    """
    ref_arch_name = work_item[0]
    try:
        r = _measure_one_worker(work_item)
        result_queue.put(r)
    except Exception:
        result_queue.put((ref_arch_name, None))


def _run_measure_with_timeout(work_item: tuple, timeout: int) -> tuple:
    """
    Run _measure_one_worker in a subprocess with timeout.
    If timeout, terminate the process and return (ref_arch_name, None).
    """
    ref_arch_name = work_item[0]
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    p = ctx.Process(target=_measure_worker_with_queue, args=(work_item, result_queue))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join(timeout=5)
        print(f"[WARNING] Measurement TIMED OUT ({timeout}s) for {ref_arch_name}, skipping")
        return (ref_arch_name, None)
    try:
        return result_queue.get_nowait()
    except Empty:
        return (ref_arch_name, None)


def record_baseline_times_resume(
    use_torch_compile: bool = False,
    torch_compile_backend: str = "inductor",
    torch_compile_options: str = "default",
    file_name: str = "baseline_time.json",
    precision: str = "fp32",
    num_gpus: int = 1,
    timeout: int = 0,
):
    """
    Generate baseline time for KernelBench with resume support.
    Loads existing results, skips problems with valid results, only measures failed/missing.
    When num_gpus > 1, runs measurements in parallel across GPUs (batch size = num_gpus).
    When timeout > 0, each operator measurement is limited to timeout seconds; hung process is terminated.
    """
    num_gpus = max(1, min(num_gpus, torch.cuda.device_count() if torch.cuda.is_available() else 1))
    save_path = os.path.join(TIMING_DIR, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load existing results
    json_results = _load_existing_results(save_path)
    if not json_results:
        json_results = {}

    measure_kwargs = (
        use_torch_compile,
        torch_compile_backend,
        torch_compile_options,
        precision,
    )

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
                to_measure.append((ref_arch_name, ref_arch_src))

        if not to_measure:
            print(f"[{level_key}] All {total} problems already have valid results, skipping.")
            continue

        timeout_str = f", timeout={timeout}s" if timeout > 0 else ""
        print(
            f"[{level_key}] {len(to_measure)}/{total} problems need measurement (resuming, num_gpus={num_gpus}{timeout_str})"
        )

        # Build work items: (ref_arch_name, ref_arch_src, device_id, *measure_kwargs)
        work_items = [
            (ref_arch_name, ref_arch_src, i % num_gpus, *measure_kwargs)
            for i, (ref_arch_name, ref_arch_src) in enumerate(to_measure)
        ]

        if timeout > 0:
            # Use per-process timeout: each task runs in its own process, we terminate on timeout
            results_list = []
            lock = threading.Lock()

            def _process_task(task_idx):
                work_item = work_items[task_idx]
                result = _run_measure_with_timeout(work_item, timeout)
                with lock:
                    results_list.append(result)

            for i in tqdm(
                range(0, len(work_items), num_gpus),
                desc=f"Level {level}",
                total=(len(work_items) + num_gpus - 1) // num_gpus,
            ):
                batch_indices = list(range(i, min(i + num_gpus, len(work_items))))
                threads = [
                    threading.Thread(target=_process_task, args=(idx,))
                    for idx in batch_indices
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                for ref_arch_name, runtime_stats in results_list[-len(batch_indices) :]:
                    json_results[level_key][ref_arch_name] = runtime_stats
                with open(save_path, "w") as f:
                    json.dump(json_results, f, indent=4)
        elif num_gpus <= 1:
            device = torch.device("cuda:0")
            for ref_arch_name, ref_arch_src in tqdm(to_measure, desc=f"Level {level}"):
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
                with open(save_path, "w") as f:
                    json.dump(json_results, f, indent=4)
        else:
            # No timeout: use Pool (original behavior)
            ctx = mp.get_context("spawn")
            with ctx.Pool(num_gpus) as pool:
                for i in tqdm(
                    range(0, len(work_items), num_gpus),
                    desc=f"Level {level}",
                    total=(len(work_items) + num_gpus - 1) // num_gpus,
                ):
                    batch = work_items[i : i + num_gpus]
                    results = pool.map(_measure_one_worker, batch)
                    for ref_arch_name, runtime_stats in results:
                        json_results[level_key][ref_arch_name] = runtime_stats
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
        precision="fp32",
    )

    print(f"Execution time for {ref_arch_name}: {exec_stats}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate baseline time with resume and optional multi-GPU parallel."
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel measurement (default: 1).",
    )
    parser.add_argument(
        "--hardware_name",
        type=str,
        default="H200",
        help="Hardware name for output directory under results/timing/ (default: H200).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Precision for baseline measurement (default: fp32).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-operator timeout in seconds. If exceeded, terminate the process and record as failed. 0 = no timeout (default).",
    )
    args = parser.parse_args()

    hardware_name = args.hardware_name
    num_gpus = args.num_gpus
    precision = args.precision
    timeout = max(0, args.timeout)

    if torch.cuda.is_available():
        n_dev = torch.cuda.device_count()
        if num_gpus > n_dev:
            print(f"[WARNING] num_gpus={num_gpus} > available {n_dev}, using num_gpus={n_dev}")
            num_gpus = n_dev
    else:
        if num_gpus > 1:
            print("[WARNING] CUDA not available, using num_gpus=1")
        num_gpus = 1

    timeout_str = f", timeout={timeout}s" if timeout > 0 else ""
    input(
        f"You are about to start recording baseline time for {hardware_name} (with resume, num_gpus={num_gpus}{timeout_str}). "
        f"Press Enter to continue..."
    )

    save_dir = os.path.join(TIMING_DIR, hardware_name)
    if os.path.exists(save_dir):
        print(
            f"📁 Found existing results in {save_dir}. Will resume - only measure failed/missing problems."
        )

    # 1. Record Torch Eager
    record_baseline_times_resume(
        use_torch_compile=False,
        torch_compile_backend=None,
        torch_compile_options=None,
        file_name=f"{hardware_name}/baseline_time_torch.json",
        precision=precision,
        num_gpus=num_gpus,
        timeout=timeout,
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
            precision=precision,
            num_gpus=num_gpus,
            timeout=timeout,
        )

    # 3. Record Torch Compile using cudagraphs
    record_baseline_times_resume(
        use_torch_compile=True,
        torch_compile_backend="cudagraphs",
        torch_compile_options=None,
        file_name=f"{hardware_name}/baseline_time_torch_compile_cudagraphs.json",
        precision=precision,
        num_gpus=num_gpus,
        timeout=timeout,
    )

    print(f"\n✅ Baseline time saved to {save_dir}")
