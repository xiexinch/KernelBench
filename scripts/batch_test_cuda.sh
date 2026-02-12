#!/bin/bash
# 批量编译并运行 cuda_optim 下所有任务，记录编译/运行结果
# 用法: bash scripts/batch_test_cuda.sh
# 结果输出到 scripts/batch_test_results.txt

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TASK_BASE="$PROJECT_ROOT/cuda_optim/kernelbench"
RESULT_FILE="$SCRIPT_DIR/batch_test_results.txt"
BINARY_NAME="test_cuda_tmp"
NVCC="nvcc"
NVCC_FLAGS="-arch=sm_89"  # RTX 4060 Laptop = sm_89
TIMEOUT_SEC=120            # 单个任务运行超时时间（秒）

# 统计计数器
total=0
compile_ok=0
compile_fail=0
run_ok=0
run_fail=0
run_timeout=0

# 清空结果文件，写入表头
cat > "$RESULT_FILE" << 'EOF'
================================================================================
  CUDA 批量编译运行测试报告
================================================================================
EOF
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULT_FILE"
echo "编译器: $NVCC" >> "$RESULT_FILE"
echo "编译参数: $NVCC_FLAGS" >> "$RESULT_FILE"
echo "运行超时: ${TIMEOUT_SEC}s" >> "$RESULT_FILE"
echo "================================================================================" >> "$RESULT_FILE"
printf "%-60s %-15s %-s\n" "任务名称" "编译状态" "运行状态" >> "$RESULT_FILE"
echo "--------------------------------------------------------------------------------" >> "$RESULT_FILE"

# 遍历所有任务目录（排除 generate_tasks.py）
for task_dir in "$TASK_BASE"/level*_run; do
    [ -d "$task_dir" ] || continue

    task_name="$(basename "$task_dir")"
    inner_dir="$task_dir/$task_name"
    src_file="$inner_dir/src/tmp_test.cu"
    inc_dir="$inner_dir/inc"

    # 检查源文件是否存在
    if [ ! -f "$src_file" ]; then
        printf "%-60s %-15s %-s\n" "$task_name" "SKIP(无源文件)" "-" >> "$RESULT_FILE"
        continue
    fi

    total=$((total + 1))
    binary_path="$inner_dir/$BINARY_NAME"

    echo "[$total] 编译: $task_name ..."

    # ============ 编译阶段 ============
    compile_log="$inner_dir/compile.log"
    $NVCC $NVCC_FLAGS -I "$inc_dir" "$src_file" -o "$binary_path" > "$compile_log" 2>&1
    compile_exit=$?

    if [ $compile_exit -ne 0 ]; then
        compile_fail=$((compile_fail + 1))
        printf "%-60s %-15s %-s\n" "$task_name" "COMPILE_ERROR" "-" >> "$RESULT_FILE"
        # 记录详细编译错误到单独的块
        {
            echo ""
            echo "  >>> 编译错误详情: $task_name"
            head -30 "$compile_log"
            echo "  <<<"
        } >> "$RESULT_FILE"
        continue
    fi

    compile_ok=$((compile_ok + 1))

    # ============ 运行阶段 ============
    echo "[$total] 运行: $task_name ..."
    run_log="$inner_dir/run.log"
    timeout "$TIMEOUT_SEC" "$binary_path" > "$run_log" 2>&1
    run_exit=$?

    if [ $run_exit -eq 124 ]; then
        # timeout 返回 124 表示超时
        run_timeout=$((run_timeout + 1))
        printf "%-60s %-15s %-s\n" "$task_name" "OK" "TIMEOUT(${TIMEOUT_SEC}s)" >> "$RESULT_FILE"
    elif [ $run_exit -ne 0 ]; then
        run_fail=$((run_fail + 1))
        printf "%-60s %-15s %-s\n" "$task_name" "OK" "RUNTIME_ERROR(exit=$run_exit)" >> "$RESULT_FILE"
        # 记录详细运行错误
        {
            echo ""
            echo "  >>> 运行错误详情: $task_name (exit=$run_exit)"
            tail -20 "$run_log"
            echo "  <<<"
        } >> "$RESULT_FILE"
    else
        run_ok=$((run_ok + 1))
        # 提取关键输出指标
        precision=$(grep -o '<precision>[^<]*</precision>' "$run_log" | sed 's/<[^>]*>//g')
        ratio=$(grep -o '<runtime_ratio>[^<]*</runtime_ratio>' "$run_log" | sed 's/<[^>]*>//g')
        time_before=$(grep -o '<time_before_opt>[^<]*</time_before_opt>' "$run_log" | sed 's/<[^>]*>//g')
        time_after=$(grep -o '<time_after_opt>[^<]*</time_after_opt>' "$run_log" | sed 's/<[^>]*>//g')

        detail="precision=$precision ratio=$ratio before=$time_before after=$time_after"
        printf "%-60s %-15s %-s\n" "$task_name" "OK" "RUN_OK ($detail)" >> "$RESULT_FILE"
    fi

    # 清理二进制文件
    rm -f "$binary_path"

done

# ============ 写入汇总 ============
{
    echo ""
    echo "================================================================================"
    echo "  汇总"
    echo "================================================================================"
    echo "总任务数:       $total"
    echo "编译成功:       $compile_ok"
    echo "编译失败:       $compile_fail"
    echo "运行成功:       $run_ok"
    echo "运行失败:       $run_fail"
    echo "运行超时:       $run_timeout"
    echo "================================================================================"
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$RESULT_FILE"

# 终端也输出汇总
echo ""
echo "========================================"
echo "  测试完成"
echo "========================================"
echo "总任务数:       $total"
echo "编译成功:       $compile_ok"
echo "编译失败:       $compile_fail"
echo "运行成功:       $run_ok"
echo "运行失败:       $run_fail"
echo "运行超时:       $run_timeout"
echo "========================================"
echo "详细报告: $RESULT_FILE"
