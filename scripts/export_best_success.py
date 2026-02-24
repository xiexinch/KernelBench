"""
从 eval_results.json 计算 best_success_file 与 failed_problem，
并将 best_success 的 kernel 文件从指定 run 目录复制到单独的输出目录。
可将 failed_problem 列表写入文件，供 generate_samples_all_levels 的 problem_subset_file 使用。
"""

import argparse
import json
import math
import os
import shutil


def load_eval_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_best_success_and_failed(rs: dict) -> tuple[dict, list[str]]:
    """
    与 read.ipynb 中逻辑一致：
    - best_success_file[level][problem] = 该题最快正确通过的 kernel 文件名，无则为 ''
    - failed_problem = 无任何正确通过的题目 ID 列表，格式 'level_<level>_problem_<id>'
    """
    best_success_file = {}
    failed_problem = []
    for level in rs.keys():
        best_success_file[level] = {}
        for problem in rs[level].keys():
            if problem not in best_success_file[level]:
                best_success_file[level][problem] = ""
            run_time = math.inf
            for sample in rs[level][problem]:
                if (
                    sample.get("compiled") is True
                    and sample.get("correctness") is True
                ):
                    r = sample.get("runtime")
                    if r is not None and r > 0 and r <= run_time:
                        best_success_file[level][problem] = (
                            f"level_{level}_problem_{problem}_sample_{sample['sample_id']}_kernel.py"
                        )
                        run_time = r
            if best_success_file[level][problem] == "":
                failed_problem.append(f"level_{level}_problem_{problem}")
    return best_success_file, failed_problem


def export_best_success(
    eval_results_path: str,
    run_dir: str,
    output_dir: str,
    save_failed: bool = False,
    failed_output_path: str | None = None,
) -> None:
    rs = load_eval_results(eval_results_path)
    best_success_file, failed_problem = compute_best_success_and_failed(rs)

    os.makedirs(output_dir, exist_ok=True)
    copied = 0
    missing = []
    for level in best_success_file:
        for problem, filename in best_success_file[level].items():
            if not filename:
                continue
            src = os.path.join(run_dir, filename)
            dst = os.path.join(output_dir, filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing.append(src)

    print(f"已复制 {copied} 个 best_success kernel 到 {output_dir}")
    if missing:
        print(f"以下 {len(missing)} 个文件在 run_dir 中不存在（已跳过）：")
        for p in missing[:10]:
            print(f"  {p}")
        if len(missing) > 10:
            print(f"  ... 共 {len(missing)} 个")

    if save_failed or failed_output_path is not None:
        out_path = failed_output_path or os.path.join(output_dir, "failed_problem.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in failed_problem:
                f.write(item + "\n")
        print(f"已写入 failed_problem 列表（共 {len(failed_problem)} 项）到 {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="从 eval_results 导出 best_success kernel 到指定文件夹，并可保存 failed_problem 列表。"
    )
    parser.add_argument(
        "--eval_results",
        default=None,
        help="eval_results.json 路径；未指定时使用 --run_dir 内的 eval_results.json",
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="存放生成 kernel 的 run 目录，例如 runs/my_run",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="导出 best_success kernel 的输出目录",
    )
    parser.add_argument(
        "--save_failed",
        action="store_true",
        help="将 failed_problem 列表写入 output_dir/failed_problem.txt",
    )
    parser.add_argument(
        "--failed_output",
        default=None,
        help="failed_problem 列表输出路径（未指定且 --save_failed 时使用 output_dir/failed_problem.txt）",
    )
    args = parser.parse_args()

    eval_results_path = args.eval_results
    if eval_results_path is None:
        eval_results_path = os.path.join(args.run_dir, "eval_results.json")
    if not os.path.isfile(eval_results_path):
        raise FileNotFoundError(
            f"未找到 eval_results: {eval_results_path}（可通过 --eval_results 指定路径）"
        )

    export_best_success(
        eval_results_path=eval_results_path,
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        save_failed=args.save_failed,
        failed_output_path=args.failed_output,
    )


if __name__ == "__main__":
    main()
