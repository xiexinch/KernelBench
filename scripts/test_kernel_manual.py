"""
手动测试单个 Kernel 文件，查看详细 CUDA 报错
使用隔离执行环境避免 CUDA 上下文污染

用法:
    uv run scripts/test_kernel_manual.py <kernel_file_path>
    
示例:
    uv run scripts/test_kernel_manual.py runs/test/level_1_problem_45_sample_0_turn_0_kernel.py
"""

import sys
import os
import shutil

# 设置环境变量（必须在导入 torch 之前）
USER = os.environ.get('USER', 'default')
os.environ["TORCH_EXTENSIONS_DIR"] = f"/tmp/torch_extensions_{USER}"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["MAX_JOBS"] = "1"
os.environ["NVCC_APPEND_FLAGS"] = "--threads 1"

# 确保目录存在并清理锁
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
lock_file = os.path.join(os.environ["TORCH_EXTENSIONS_DIR"], "lock")
if os.path.exists(lock_file):
    try:
        os.remove(lock_file)
        print(f"[Info] 清理锁文件: {lock_file}")
    except Exception as e:
        print(f"[Warning] 无法清理锁文件: {e}")

import torch
import torch.nn as nn
import traceback
import gc

# 添加项目路径
REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_TOP_DIR, "src"))

from kernelbench.dataset import construct_kernelbench_dataset


def cleanup_all(device=None):
    """彻底清理 CUDA 资源"""
    gc.collect()
    if torch.cuda.is_available():
        if device is not None:
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()


def scale_down_inputs(input_args, max_batch=2, max_channels=8, max_spatial=256):
    """
    减小输入尺寸以避免 OOM
    
    例如将 [16, 64, 2048, 2048] 减小为 [2, 8, 256, 256]
    """
    scaled_input_args = []
    for x in input_args:
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            # 4D 张量 (batch, channels, height, width)
            b = min(x.size(0), max_batch)
            c = min(x.size(1), max_channels)
            h = min(x.size(2), max_spatial)
            w = min(x.size(3), max_spatial)
            x = x[:b, :c, :h, :w]
            scaled_input_args.append(x)
        elif isinstance(x, torch.Tensor) and x.dim() == 3:
            # 3D 张量
            d0 = min(x.size(0), max_batch)
            d1 = min(x.size(1), max_channels)
            d2 = min(x.size(2), max_spatial)
            x = x[:d0, :d1, :d2]
            scaled_input_args.append(x)
        elif isinstance(x, torch.Tensor) and x.dim() == 2:
            # 2D 张量
            d0 = min(x.size(0), max_spatial)
            d1 = min(x.size(1), max_spatial)
            x = x[:d0, :d1]
            scaled_input_args.append(x)
        else:
            scaled_input_args.append(x)
    return scaled_input_args


def run_model_isolated(model_class, init_args, input_args, device, seed=42):
    """
    隔离运行模型：加载 -> 前向 -> 释放 -> 清理 -> 返回 CPU 结果
    """
    model = None
    output = None
    
    try:
        # 设置随机种子
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # 创建模型
        model = model_class(*init_args)
        model = model.to(device)
        model.eval()
        
        # 准备输入（确保在正确的设备上）
        input_tensors = []
        for x in input_args:
            if isinstance(x, torch.Tensor):
                input_tensors.append(x.to(device))
            else:
                input_tensors.append(x)
        
        # 前向传播
        with torch.no_grad():
            output = model(*input_tensors)
        
        # 同步确保完成
        torch.cuda.synchronize(device)
        
        # 将结果移到 CPU（深拷贝）
        if isinstance(output, tuple):
            output = tuple(o.cpu().clone() if isinstance(o, torch.Tensor) else o for o in output)
        elif isinstance(output, torch.Tensor):
            output = output.cpu().clone()
        
        return output, True, None
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return None, False, error_msg
        
    finally:
        # 彻底清理
        del model
        if output is not None and isinstance(output, torch.Tensor):
            del output
        cleanup_all(device)


def test_kernel_file(kernel_path: str, level: int = 1, verbose: bool = True):
    """测试单个 kernel 文件，使用隔离执行避免 CUDA 上下文污染"""
    
    # 从文件名解析信息
    filename = os.path.basename(kernel_path)
    print(f"=" * 70)
    print(f"测试文件: {filename}")
    print(f"=" * 70)
    print(f"TORCH_EXTENSIONS_DIR: {os.environ.get('TORCH_EXTENSIONS_DIR')}")
    print("-" * 70)
    
    # 加载 kernel 代码
    with open(kernel_path, 'r', encoding='utf-8') as f:
        kernel_code = f.read()
    
    # 尝试从文件名解析 problem_id
    try:
        parts = filename.split('_')
        problem_id = int(parts[parts.index('problem') + 1])
    except (ValueError, IndexError):
        problem_id = int(input("请输入 problem_id: "))
    
    print(f"级别: {level}, 问题 ID: {problem_id}")
    print(f"代码长度: {len(kernel_code)} 字符")
    print("-" * 70)
    
    # 加载数据集获取参考架构
    dataset = construct_kernelbench_dataset(
        level=level,
        source="local",
        dataset_name="ScalingIntelligence/KernelBench",
    )
    problem = dataset.get_problem_by_id(problem_id)
    ref_arch_src = problem.code
    
    print(f"参考架构: {problem.name}")
    print("-" * 70)
    
    # 设置设备
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用")
        return
    
    device = torch.cuda.current_device()
    print(f"使用设备: cuda:{device}")
    print("=" * 70)
    print()
    
    # ========== 步骤 1: 加载参考架构的函数 ==========
    print("[步骤 1] 加载参考架构配置...")
    try:
        ref_context = {}
        exec(compile(ref_arch_src, "<string>", "exec"), ref_context)
        
        Model = ref_context.get("Model")
        get_init_inputs = ref_context.get("get_init_inputs")
        get_inputs = ref_context.get("get_inputs")
        
        if not all([Model, get_init_inputs, get_inputs]):
            raise ValueError("参考架构缺少必要的类或函数")
        
        # 获取初始化参数和输入（使用小尺寸）
        torch.manual_seed(42)
        init_args = get_init_inputs()
        torch.manual_seed(42)
        input_args = get_inputs()
        
        # 减小输入尺寸以避免 OOM
        input_args = scale_down_inputs(input_args)
        
        print("✓ 参考架构加载成功")
        print(f"  初始化参数: {len(init_args)} 个")
        print(f"  输入参数: {len(input_args)} 个")
        
    except Exception as e:
        print(f"✗ 参考架构加载失败: {e}")
        traceback.print_exc()
        return
    print()
    
    # ========== 步骤 2: 运行参考模型（隔离）==========
    print("[步骤 2] 运行参考模型（隔离执行）...")
    ref_output, ref_success, ref_error = run_model_isolated(
        Model, init_args, input_args, device, seed=42
    )
    
    if not ref_success:
        print(f"✗ 参考模型运行失败")
        print("=" * 70)
        print(ref_error)
        print("=" * 70)
        return
    
    print("✓ 参考模型运行成功")
    if isinstance(ref_output, torch.Tensor):
        print(f"  输出形状: {ref_output.shape}")
    elif isinstance(ref_output, tuple):
        shapes = [o.shape if isinstance(o, torch.Tensor) else type(o) for o in ref_output]
        print(f"  输出形状: {shapes}")
    print()
    
    # ========== 步骤 3: 编译自定义 Kernel ==========
    print("[步骤 3] 编译自定义 CUDA kernel...")
    print("-" * 70)
    
    # 设置编译环境
    torch.cuda.set_device(device)
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    ModelNew = None
    
    try:
        # 尝试编译 kernel
        context_new = {}
        compiled = compile(kernel_code, "<string>", "exec")
        exec(kernel_code, context_new)
        ModelNew = context_new.get("ModelNew")
        
        if ModelNew is None:
            raise ValueError("找不到 ModelNew 类")
        
        print("✓ Python 编译成功")
        print(f"  ModelNew 类: {ModelNew}")
        
    except Exception as e:
        print(f"✗ 编译失败: {e}")
        print()
        print("详细错误信息:")
        print("=" * 70)
        traceback.print_exc()
        print("=" * 70)
        
        # 分析错误类型
        error_lower = str(e).lower()
        if "nvcc" in error_lower or "compilation" in error_lower or "error:" in error_lower:
            print()
            print("[错误类型] CUDA 编译错误 (nvcc)")
            print("常见原因:")
            print("- CUDA 语法错误")
            print("- 未定义的标识符")
            print("- 类型不匹配")
        return
    
    finally:
        cleanup_all(device)
    
    print()
    
    # ========== 步骤 4: 运行自定义模型（隔离）==========
    print("[步骤 4] 运行自定义模型（隔离执行）...")
    print("-" * 70)
    
    custom_output, custom_success, custom_error = run_model_isolated(
        ModelNew, init_args, input_args, device, seed=42
    )
    
    if not custom_success:
        print(f"✗ 自定义模型运行失败")
        print("=" * 70)
        print(custom_error)
        print("=" * 70)
        
        # 分析错误类型
        if "illegal memory access" in custom_error.lower():
            print()
            print("[错误分析] CUDA 非法内存访问")
            print("可能原因:")
            print("- 数组越界访问 (检查索引计算)")
            print("- 未初始化的共享内存")
            print("- __syncthreads() 使用不当")
            print("- 线程块配置错误")
        elif "out of memory" in custom_error.lower():
            print()
            print("[错误分析] CUDA 内存不足")
        return
    
    print("✓ 自定义模型运行成功")
    if isinstance(custom_output, torch.Tensor):
        print(f"  输出形状: {custom_output.shape}")
    elif isinstance(custom_output, tuple):
        shapes = [o.shape if isinstance(o, torch.Tensor) else type(o) for o in custom_output]
        print(f"  输出形状: {shapes}")
    print()
    
    # ========== 步骤 5: 对比结果（在 CPU 上）==========
    print("[步骤 5] 对比结果（CPU 内存中）...")
    print("-" * 70)
    
    try:
        # 处理元组输出（取第一个元素对比）
        ref_tensor = ref_output[0] if isinstance(ref_output, tuple) else ref_output
        custom_tensor = custom_output[0] if isinstance(custom_output, tuple) else custom_output
        
        if not isinstance(ref_tensor, torch.Tensor) or not isinstance(custom_tensor, torch.Tensor):
            print("✗ 输出不是张量，无法对比")
            return
        
        # 在 CPU 上对比
        all_close = torch.allclose(ref_tensor, custom_tensor, rtol=1e-2, atol=1e-3)
        max_diff = (ref_tensor - custom_tensor).abs().max().item()
        mean_diff = (ref_tensor - custom_tensor).abs().mean().item()
        
        if all_close:
            print("✓ 正确性测试通过")
            print(f"  最大差异: {max_diff:.6f}")
            print(f"  平均差异: {mean_diff:.6f}")
        else:
            print("✗ 正确性测试失败")
            print(f"  最大差异: {max_diff:.6f}")
            print(f"  平均差异: {mean_diff:.6f}")
            print(f"  参考输出范围: [{ref_tensor.min():.4f}, {ref_tensor.max():.4f}]")
            print(f"  自定义输出范围: [{custom_tensor.min():.4f}, {custom_tensor.max():.4f}]")
            
            # 显示前几个不同的值
            diff_mask = (ref_tensor - custom_tensor).abs() > 1e-3
            if diff_mask.any():
                diff_indices = torch.where(diff_mask)
                num_diffs = min(5, len(diff_indices[0]))
                print(f"  前 {num_diffs} 个差异位置:")
                for i in range(num_diffs):
                    idx = tuple(idx_tensor[i].item() for idx_tensor in diff_indices)
                    ref_val = ref_tensor[idx].item()
                    custom_val = custom_tensor[idx].item()
                    print(f"    {idx}: 参考={ref_val:.6f}, 自定义={custom_val:.6f}")
                    
    except Exception as e:
        print(f"✗ 结果对比失败: {e}")
        traceback.print_exc()
    
    print()
    print("=" * 70)
    print("测试完成")
    print("=" * 70)


def interactive_test():
    """交互式测试模式"""
    print("=" * 70)
    print("KernelBench 手动测试工具（隔离执行模式）")
    print("=" * 70)
    print()
    
    # 列出可用的 kernel 文件
    runs_dir = os.path.join(REPO_TOP_DIR, "runs")
    if os.path.exists(runs_dir):
        runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if runs:
            print("可用的运行目录:")
            for i, run in enumerate(sorted(runs), 1):
                print(f"  {i}. {run}")
            print()
    
    # 获取 kernel 路径
    kernel_path = input("请输入 kernel 文件路径: ").strip()
    
    if not os.path.exists(kernel_path):
        # 尝试在 runs 目录下查找
        alt_path = os.path.join(runs_dir, kernel_path)
        if os.path.exists(alt_path):
            kernel_path = alt_path
        else:
            print(f"错误: 文件不存在: {kernel_path}")
            return
    
    # 获取级别
    level_input = input("请输入级别 (1-4, 默认 1): ").strip()
    level = int(level_input) if level_input else 1
    
    # 运行测试
    test_kernel_file(kernel_path, level=level)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 交互式模式
        interactive_test()
    else:
        # 命令行模式
        kernel_path = sys.argv[1]
        level = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        
        if not os.path.exists(kernel_path):
            print(f"错误: 文件不存在: {kernel_path}")
            print(f"用法: uv run scripts/test_kernel_manual.py <kernel_file_path> [level]")
            sys.exit(1)
        
        test_kernel_file(kernel_path, level=level)
