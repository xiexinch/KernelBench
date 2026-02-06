#!/usr/bin/env python3
"""
验证 level4_extend 展开实现与 level4 原始实现（HuggingFace transformers）的精度对齐

用法:
    python scripts/verify_level4_extend_precision.py                    # 测试所有 GPT-2 文件
    python scripts/verify_level4_extend_precision.py --problem 16         # 仅测试 problem 16
    python scripts/verify_level4_extend_precision.py --device cpu        # 使用 CPU（默认 cuda）

注意: problem 19 (batch_size=1024) 在 CPU 上可能因内存不足被终止，建议用 --problem 7 或 16 快速验证。
"""

import argparse
import os
import sys

import torch

# 添加项目路径
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

KERNEL_BENCH_PATH = os.path.join(REPO_ROOT, "KernelBench")
LEVEL4_PATH = os.path.join(KERNEL_BENCH_PATH, "level4")
LEVEL4_EXTEND_PATH = os.path.join(KERNEL_BENCH_PATH, "level4_extend")

# GPT-2 对应的 problem ids
GPT2_PROBLEM_IDS = [7, 16, 19]


def load_model_from_code(code: str, context: dict):
    """从代码字符串加载 Model、get_init_inputs、get_inputs"""
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        raise RuntimeError(f"语法错误: {e}") from e
    exec(code, context)
    Model = context.get("Model")
    get_init_inputs = context.get("get_init_inputs")
    get_inputs = context.get("get_inputs")
    if Model is None or get_init_inputs is None or get_inputs is None:
        raise RuntimeError("代码中缺少 Model、get_init_inputs 或 get_inputs")
    return Model, get_init_inputs, get_inputs


def get_problem_filename(problem_id: int) -> str:
    """根据 problem_id 获取文件名"""
    for name in os.listdir(LEVEL4_PATH):
        if name.startswith(f"{problem_id}_") and name.endswith(".py"):
            return name
    raise FileNotFoundError(f"未找到 problem {problem_id} 对应的文件")


def convert_hf_gpt2_state_dict_for_expanded(hf_state_dict):
    """
    将 HuggingFace GPT2 的 state_dict 转为展开模型可用的格式。
    HF 的 Conv1D 使用 (in, out) 权重布局，nn.Linear 使用 (out, in)，需转置。
    """
    new_sd = {}
    for k, v in hf_state_dict.items():
        if "c_attn.weight" in k or "c_proj.weight" in k or "c_fc.weight" in k:
            new_sd[k] = v.T.clone()
        else:
            new_sd[k] = v
    return new_sd


def run_precision_test(problem_id: int, device: torch.device, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    """对单个 problem 进行精度对齐测试"""
    filename = get_problem_filename(problem_id)
    level4_path = os.path.join(LEVEL4_PATH, filename)
    level4_extend_path = os.path.join(LEVEL4_EXTEND_PATH, filename)

    if not os.path.exists(level4_extend_path):
        print(f"  [SKIP] level4_extend 中无对应文件: {filename}")
        return True

    with open(level4_path, "r", encoding="utf-8") as f:
        level4_code = f.read()
    with open(level4_extend_path, "r", encoding="utf-8") as f:
        level4_extend_code = f.read()

    # 加载原始模型（依赖 transformers）
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        print(f"  [SKIP] 需要安装 transformers: pip install transformers")
        return True

    context_orig = {}
    Model_orig, get_init_inputs_orig, get_inputs_orig = load_model_from_code(level4_code, context_orig)

    context_ext = {}
    Model_ext, get_init_inputs_ext, get_inputs_ext = load_model_from_code(level4_extend_code, context_ext)

    # 固定随机种子以保证输入一致
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)

    # 原始模型：init_inputs = (model_name, config)
    init_orig = get_init_inputs_orig()
    model_orig = Model_orig(*init_orig)
    model_orig = model_orig.to(device)
    model_orig.eval()

    # 展开模型：init_inputs = (config,)
    init_ext = get_init_inputs_ext()
    model_ext = Model_ext(*init_ext)
    model_ext = model_ext.to(device)
    model_ext.eval()

    # 将原始模型的权重加载到展开模型
    # HF Conv1D 与 nn.Linear 权重布局不同，需转置
    hf_state = model_orig.model.state_dict()
    expanded_state = convert_hf_gpt2_state_dict_for_expanded(hf_state)
    load_result = model_ext.model.load_state_dict(expanded_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        # 可能因 tied weights 等有少量差异，先尝试运行
        if load_result.missing_keys:
            print(f"  [WARN] 缺失的 keys: {load_result.missing_keys[:5]}...")
        if load_result.unexpected_keys:
            print(f"  [WARN] 未预期的 keys: {load_result.unexpected_keys[:5]}...")

    # 生成相同输入
    inputs = get_inputs_orig()
    input_ids = inputs[0].to(device)

    with torch.no_grad():
        out_orig = model_orig(input_ids)
        out_ext = model_ext(input_ids)

    # 比较输出
    if out_orig.shape != out_ext.shape:
        print(f"  [FAIL] 输出形状不一致: {out_orig.shape} vs {out_ext.shape}")
        return False

    # 使用 allclose 检查数值对齐
    if torch.allclose(out_orig, out_ext, atol=atol, rtol=rtol):
        max_diff = (out_orig - out_ext).abs().max().item()
        print(f"  [PASS] 精度对齐 (max_diff={max_diff:.2e})")
        return True
    else:
        max_diff = (out_orig - out_ext).abs().max().item()
        mean_diff = (out_orig - out_ext).abs().mean().item()
        print(f"  [FAIL] 精度未对齐 (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
        return False


def main():
    parser = argparse.ArgumentParser(description="验证 level4_extend 与 level4 的精度对齐")
    parser.add_argument("--problem", type=int, choices=GPT2_PROBLEM_IDS, default=None,
                        help="指定 problem id，不指定则测试全部")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备 (cuda/cpu)")
    parser.add_argument("--atol", type=float, default=1e-4, help="绝对误差容限")
    parser.add_argument("--rtol", type=float, default=1e-4, help="相对误差容限")
    args = parser.parse_args()

    device = torch.device(args.device)
    problem_ids = [args.problem] if args.problem is not None else GPT2_PROBLEM_IDS

    print("=" * 60)
    print("Level4 Extend 精度对齐验证")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"测试 problems: {problem_ids}")
    print()

    passed = 0
    failed = 0
    for pid in problem_ids:
        print(f"Problem {pid} ({get_problem_filename(pid)}):")
        try:
            ok = run_precision_test(pid, device, atol=args.atol, rtol=args.rtol)
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
