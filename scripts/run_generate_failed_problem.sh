#!/usr/bin/env bash
# 使用 failed_problem 子集调用 scripts/generate_samples_all_levels.py
# 用法:
#   bash scripts/run_generate_failed_problem.sh <failed_problem_file> [其他 pydra 参数...]
# 示例:
#   bash scripts/run_generate_failed_problem.sh failed_problem.txt dataset_src=local run_name=retry_failed server_type=...
#   bash scripts/run_generate_failed_problem.sh ./out/failed_problem.txt dataset_src=local run_name=retry include_level4_expand=true

set -e
REPO_TOP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_TOP"

FAILED_FILE="$1"
if [[ -z "$FAILED_FILE" ]]; then
  echo "用法: $0 <failed_problem_file> [generate_samples_all_levels.py 的其余参数...]"
  echo "  failed_problem_file: 每行一个 level_<level>_problem_<id>，可由 export_best_success.py --save_failed 生成"
  exit 1
fi
shift

if [[ ! -f "$FAILED_FILE" ]]; then
  echo "错误: 文件不存在: $FAILED_FILE"
  exit 1
fi

echo "使用 failed_problem 子集: $FAILED_FILE"
exec uv run python scripts/generate_samples_all_levels.py problem_subset_file="$FAILED_FILE" "$@"
