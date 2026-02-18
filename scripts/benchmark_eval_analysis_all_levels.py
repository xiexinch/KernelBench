"""
Analysis for all levels from a single multi-level eval_results.json.
Outputs one CSV with per-level rows and an Overall row. Each script is independent.
"""

import csv
import json
import os
from typing import Optional

import numpy as np
import pydra
from pydra import Config, REQUIRED

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.score import (
    fastp,
    geometric_mean_speed_ratio_correct_only,
)

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 benchmark_eval_analysis 一致的 p 阈值
P_VALUES = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]


def level_key_to_dataset_spec(level_key: str) -> tuple[int, Optional[str]]:
    """level_key -> (dataset_level, local_subdir) 用于加载 dataset。"""
    if level_key == "level4_expand":
        return 4, "level4_expand"
    if level_key.startswith("level") and level_key[5:].isdigit():
        return int(level_key[5:]), None
    raise ValueError(f"Unknown level_key: {level_key}")


def get_level_keys_from_eval_results(eval_results: dict) -> list[str]:
    """从多 level eval_results 的顶层 key 得到 level 列表（仅以 level 开头的 key，按 level1,2,3,4, level4_expand 顺序）。"""
    order = ["level1", "level2", "level3", "level4", "level4_expand"]
    keys = [k for k in order if k in eval_results]
    # 若有其他 level 键也加入
    for k in sorted(eval_results.keys()):
        if k.startswith("level") and k not in keys:
            keys.append(k)
    return keys


def patch(eval_results: dict, dataset) -> dict:
    """对单 level 的 eval_results 按 dataset 补全缺失的 problem_id。"""
    for pid in dataset.get_problem_ids():
        if str(pid) not in eval_results:
            eval_results[str(pid)] = {
                "sample_id": 0,
                "compiled": False,
                "correctness": False,
                "metadata": {},
                "runtime": -1.0,
                "runtime_stats": {},
            }
    return eval_results


def analyze_one_level(
    level_key: str,
    level_eval: dict,
    baseline_results: dict,
) -> tuple[dict, list, list, list]:
    """
    分析单 level，返回 (metrics_dict, is_correct_list, baseline_speed_list, actual_speed_list)
    用于后续合并计算 Overall。
    """
    dataset_level, local_subdir = level_key_to_dataset_spec(level_key)
    dataset = construct_kernelbench_dataset(
        level=dataset_level,
        source="local",
        local_subdir=local_subdir,
    )
    # 只取 sample_id == 0（与 benchmark_eval_analysis 一致）
    stripped = {}
    for key, result in level_eval.items():
        if isinstance(result, list):
            entry = [r for r in result if r.get("sample_id") == 0]
        else:
            entry = [result] if result.get("sample_id") == 0 else []
        if len(entry) == 1:
            stripped[key] = entry[0]
    level_eval = patch(stripped, dataset)

    total_count = len(dataset)
    compiled_count = sum(1 for e in level_eval.values() if e.get("compiled"))
    correct_count = sum(1 for e in level_eval.values() if e.get("correctness"))

    is_correct_list = []
    baseline_speed_list = []
    actual_speed_list = []
    sorted_pids = sorted(dataset.get_problem_ids())

    for pid in sorted_pids:
        if str(pid) not in level_eval:
            continue
        eval_entry = level_eval[str(pid)]
        problem = dataset.get_problem_by_id(pid)
        problem_name = problem.name
        if level_key not in baseline_results or problem_name not in baseline_results[level_key]:
            continue
        baseline_entry = baseline_results[level_key][problem_name]
        if baseline_entry is None:
            continue
        is_correct_list.append(eval_entry["correctness"])
        actual_speed_list.append(eval_entry["runtime"])
        baseline_speed_list.append(baseline_entry["mean"])

    n = len(is_correct_list)
    if n == 0:
        geo_mean = 0.0
        fast_p_dict = {str(p): 0.0 for p in P_VALUES}
    else:
        is_correct = np.array(is_correct_list)
        baseline_speed = np.array(baseline_speed_list)
        actual_speed = np.array(actual_speed_list)
        geo_mean = geometric_mean_speed_ratio_correct_only(
            is_correct, baseline_speed, actual_speed, n
        )
        fast_p_dict = {
            str(p): float(fastp(is_correct, baseline_speed, actual_speed, n, p))
            for p in P_VALUES
        }

    metrics = {
        "level": level_key,
        "total_count": total_count,
        "compiled_count": compiled_count,
        "correct_count": correct_count,
        "compilation_rate": compiled_count / total_count if total_count > 0 else 0.0,
        "correctness_rate": correct_count / total_count if total_count > 0 else 0.0,
        "geo_mean_speedup": geo_mean,
        "fast_p": fast_p_dict,
    }
    return metrics, is_correct_list, baseline_speed_list, actual_speed_list


class AnalysisAllLevelsConfig(Config):
    """Config for analyzing all levels; no level parameter."""

    def __init__(self):
        self.run_name = REQUIRED
        self.hardware = REQUIRED
        self.baseline = REQUIRED
        self.eval_results_dir = None
        self.baseline_file = None
        self.output_csv = None  # 输出 CSV 路径，默认 runs/{run_name}/analysis_all_levels.csv

    def __repr__(self):
        return f"AnalysisAllLevelsConfig({self.to_dict()})"


@pydra.main(base=AnalysisAllLevelsConfig)
def main(config: AnalysisAllLevelsConfig):
    """一次计算所有 level 的统计，分 level 与 Overall 写入一个 CSV。"""
    if config.eval_results_dir:
        eval_file_path = os.path.join(
            config.eval_results_dir, config.run_name, "eval_results.json"
        )
    else:
        eval_file_path = os.path.join(
            REPO_TOP_DIR, "runs", config.run_name, "eval_results.json"
        )
    if not os.path.exists(eval_file_path):
        raise FileNotFoundError(
            f"Eval results not found at {eval_file_path}. Run eval_from_generations_all_levels.py first."
        )

    if config.baseline_file:
        baseline_file_path = config.baseline_file
    else:
        baseline_file_path = os.path.join(
            REPO_TOP_DIR,
            "results",
            "timing",
            config.hardware,
            f"{config.baseline}.json",
        )
    if not os.path.exists(baseline_file_path):
        raise FileNotFoundError(
            f"Baseline file not found at {baseline_file_path}"
        )

    with open(eval_file_path, "r") as f:
        eval_results = json.load(f)
    with open(baseline_file_path, "r") as f:
        baseline_results = json.load(f)

    level_keys = get_level_keys_from_eval_results(eval_results)
    if not level_keys:
        raise ValueError(
            "eval_results.json 中未找到 level 键（level1, level2, ...）。请使用 eval_from_generations_all_levels.py 生成多 level 结果。"
        )

    eval_results_dir = config.eval_results_dir or os.path.join(REPO_TOP_DIR, "runs")
    rows = []
    all_correct = []
    all_baseline_speed = []
    all_actual_speed = []
    total_count_sum = 0
    compiled_sum = 0
    correct_sum = 0

    for level_key in level_keys:
        if level_key not in eval_results:
            continue
        level_eval = eval_results[level_key]
        metrics, ic, bs, as_ = analyze_one_level(
            level_key,
            level_eval,
            baseline_results,
        )
        rows.append(metrics)
        total_count_sum += metrics["total_count"]
        compiled_sum += metrics["compiled_count"]
        correct_sum += metrics["correct_count"]
        all_correct.extend(ic)
        all_baseline_speed.extend(bs)
        all_actual_speed.extend(as_)

    # Overall 行
    n_all = len(all_correct)
    if n_all > 0:
        is_correct = np.array(all_correct)
        baseline_speed = np.array(all_baseline_speed)
        actual_speed = np.array(all_actual_speed)
        geo_overall = geometric_mean_speed_ratio_correct_only(
            is_correct, baseline_speed, actual_speed, n_all
        )
        fast_p_overall = {
            str(p): float(fastp(is_correct, baseline_speed, actual_speed, n_all, p))
            for p in P_VALUES
        }
    else:
        geo_overall = 0.0
        fast_p_overall = {str(p): 0.0 for p in P_VALUES}

    overall_row = {
        "level": "Overall",
        "total_count": total_count_sum,
        "compiled_count": compiled_sum,
        "correct_count": correct_sum,
        "compilation_rate": compiled_sum / total_count_sum if total_count_sum > 0 else 0.0,
        "correctness_rate": correct_sum / total_count_sum if total_count_sum > 0 else 0.0,
        "geo_mean_speedup": geo_overall,
        "fast_p": fast_p_overall,
    }
    rows.append(overall_row)

    # 写 CSV
    if config.output_csv:
        csv_path = config.output_csv
    else:
        csv_path = os.path.join(
            eval_results_dir, config.run_name, "analysis_all_levels.csv"
        )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Level",
            "TotalCount",
            "CompiledCount",
            "CorrectCount",
            "CompilationRate",
            "CorrectnessRate",
            "GeoMeanSpeedup",
            "Fast_p_0", "Fast_p_0.5", "Fast_p_0.8", "Fast_p_1.0", "Fast_p_1.5", "Fast_p_2.0",
        ])
        for r in rows:
            writer.writerow([
                r["level"],
                r["total_count"],
                r["compiled_count"],
                r["correct_count"],
                f"{r['compilation_rate']:.4f}",
                f"{r['correctness_rate']:.4f}",
                f"{r['geo_mean_speedup']:.4f}",
                *[f"{r['fast_p'][str(p)]:.4f}" for p in P_VALUES],
            ])
    print(f"Results written to: {csv_path}")


if __name__ == "__main__":
    main()
