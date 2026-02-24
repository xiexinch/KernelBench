"""
Multi-turn batch generation for all KernelBench levels (Kevin-style).

Reference workflow: generate -> execute/evaluate -> feedback -> refine, for multiple turns.
This script mirrors `scripts/generate_samples_all_levels.py` (all levels + resume) but
adds multi-turn refinement per (level, problem, sample).

LLM 请求经 create_inference_server_from_presets -> query_server。通过网关访问 vLLM 时，
在 .env 中设置 HOSTED_VLLM_API_BASE、HOSTED_VLLM_HOST（及可选 HOSTED_VLLM_API_KEY）
即可自动带上 Host 头与 api_base；server_type=local 时可通过 LOCAL_SERVER_HOST 指定 Host。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import os
import sys
import time

import pydra

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string, KernelExecResult
from kernelbench.kernel_static_checker import validate_kernel_static
from kernelbench.prompt_constructor_toml import get_custom_prompt, get_prompt_for_backend
from kernelbench.utils import create_inference_server_from_presets, extract_first_code, maybe_multithread

from generate_samples import GenerationConfig, WorkArgs, check_kernel_exists


REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_all_level_specs(
    dataset_src: str, include_level4_expand: bool
) -> list[tuple[int, Optional[str], str]]:
    """
    返回要跑的所有 level 的 (dataset_level, local_subdir, level_label) 列表。
    level_label 用于文件命名，如 level_1、level_level4_expand。
    """
    specs: list[tuple[int, Optional[str], str]] = [
        (1, None, "1"),
        (2, None, "2"),
        (3, None, "3"),
        (4, None, "4"),
    ]
    if include_level4_expand and dataset_src == "local":
        specs.append((4, "level4_expand", "level4_expand"))
    elif include_level4_expand and dataset_src != "local":
        raise ValueError("level4_expand is only supported with dataset_src=local")
    return specs


def _run_dir_has_existing_kernels(run_dir: str) -> bool:
    """检查 run 目录下是否已有生成的 kernel 文件（用于判断是否为中断后重跑）。"""
    if not os.path.isdir(run_dir):
        return False
    for name in os.listdir(run_dir):
        if name.endswith("_kernel.py") and name.startswith("level_"):
            return True
    return False


def _ask_resume(run_dir: str) -> bool:
    """
    当检测到已有结果时询问用户是否 resume（跳过已生成、只补全未生成的）。
    返回 True 表示 resume，False 表示不 resume（用户应使用新 run_name 重新跑）。
    非交互环境（非 TTY）下默认 resume，不阻塞。
    """
    if not sys.stdin.isatty():
        print(
            f"Run directory already exists with existing kernels: {run_dir}. "
            "Non-interactive mode: resuming (skip existing, generate missing)."
        )
        return True
    print(f"\n⚠️  Run directory already exists: {run_dir}")
    print("   Previous run may have been interrupted. Existing kernels will be skipped if you resume.")
    while True:
        try:
            answer = input("   Resume (skip existing, only generate missing)? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)
        if answer in ("", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("   Please enter Y or n.")


class GenerationMultiturnAllLevelsConfig(GenerationConfig):
    """
    Config for multi-turn generation across all levels.

    Extends `GenerationConfig` and fixes `level` to 'all'.
    """

    def __init__(self):
        super().__init__()
        self.level = "all"
        self.include_level4_expand = False

        # Multi-turn refinement
        self.max_turns = 4
        self.early_stop_on_correct = True

        # 是否将多轮对话保存为 JSON（messages: system/user/assistant 交替）
        self.log_conversation = True
        # 写入 conversation JSON 时的 system 消息内容，可为空
        self.conversation_system_prompt = ""

        # Generating + evaluating kernels is GPU-heavy; default to 1 worker for safety.
        self.num_workers = 1


@dataclass
class AttemptRecord:
    kernel_code: str
    summary: Optional[str]
    eval_result: KernelExecResult


def _extract_summary_after_first_codeblock(raw: str) -> Optional[str]:
    """
    Best-effort: return trailing text after the first ```...``` block as summary.
    Keeps it short to avoid context blow-up.
    """
    if not raw:
        return None
    s = raw.strip()
    start = s.find("```")
    if start < 0:
        return None
    end = s.find("```", start + 3)
    if end < 0:
        return None
    end2 = s.find("```", end + 3)
    # If a second fence exists, use the end of the first fenced block as the boundary:
    boundary = end2 if end2 >= 0 else end + 3
    tail = s[boundary:].strip()
    if not tail:
        return None
    # Truncate to keep prompts bounded
    return tail[:1200]


def _make_parsing_failure_result(message: str) -> KernelExecResult:
    return KernelExecResult(compiled=False, correctness=False, metadata={"parsing_error": message})


def _eval_result_to_turn_metric(turn_index: int, result: KernelExecResult) -> dict:
    """将 KernelExecResult 转为可写入 JSON 的一轮性能摘要。"""
    md = result.metadata or {}
    speedup = None
    ref_runtime_us = None
    runtime_us = None
    if result.runtime > 0 and result.ref_runtime > 0:
        runtime_us = round(result.runtime, 4)
        ref_runtime_us = round(result.ref_runtime, 4)
        speedup = round(float(result.ref_runtime) / float(result.runtime), 4)
    error = None
    if not result.compiled:
        error = str(md.get("compilation_error") or md.get("parsing_error") or md.get("compilation_error_name", "Unknown"))
    elif not result.correctness:
        error = str(md.get("runtime_error") or md.get("correctness_issue") or "Unknown")
    return {
        "turn": turn_index,
        "compiled": result.compiled,
        "correctness": result.correctness,
        "speedup": speedup,
        "ref_runtime_us": ref_runtime_us,
        "runtime_us": runtime_us,
        "error": error,
    }


def _turn_metric_to_eval_result(metric: dict) -> KernelExecResult:
    """从 JSON 中的 turn_metric 恢复 KernelExecResult（用于 resume）。"""
    runtime = metric.get("runtime_us")
    ref_runtime = metric.get("ref_runtime_us")
    return KernelExecResult(
        compiled=bool(metric.get("compiled", False)),
        correctness=bool(metric.get("correctness", False)),
        runtime=float(runtime) if runtime is not None else -1.0,
        ref_runtime=float(ref_runtime) if ref_runtime is not None else -1.0,
        metadata={"error": metric.get("error")} if metric.get("error") else {},
    )


def _write_conversation_json(
    run_dir: str,
    level_label: str,
    problem_id: int,
    sample_id: int,
    conversation_messages: list,
    history: list,
) -> None:
    """每轮结束后立即写入对话与 turn_metrics 到 JSON。"""
    conv_path = os.path.join(
        run_dir,
        f"level_{level_label}_problem_{problem_id}_sample_{sample_id}_conversation.json",
    )
    turn_metrics = [_eval_result_to_turn_metric(i, rec.eval_result) for i, rec in enumerate(history)]
    payload = {"messages": conversation_messages, "turn_metrics": turn_metrics}
    with open(conv_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def format_feedback_from_eval_result(result: KernelExecResult) -> str:
    """
    Format feedback in a Kevin-style multi-turn message.
    """
    md = result.metadata or {}

    if md.get("parsing_error"):
        return (
            "Your previous answer failed to be parsed due to not adhering to the desired formatting. "
            "Here is the error message:\n"
            f"{md.get('parsing_error')}"
        )

    if not result.compiled:
        err = md.get("compilation_error")
        return (
            "Your previous answer failed to compile. Here is the error message:\n"
            f"{err if err is not None else md.get('compilation_error_name', 'Unknown compilation error')}"
        )

    if not result.correctness:
        err = md.get("runtime_error") or md.get("correctness_issue") or "Unknown runtime/correctness error"
        return (
            "Your previous answer compiled successfully but had runtime/correctness errors. "
            "Here is the error message:\n"
            f"{err}"
        )

    # Correct case
    speedup_str = "N/A"
    try:
        if result.runtime > 0 and result.ref_runtime > 0:
            speedup = float(result.ref_runtime) / float(result.runtime)
            speedup_str = f"{speedup:.4f}x"
    except Exception:
        pass

    return (
        "Your previous answer was correct but can be made faster. "
        f"Here is the speedup you achieved relative to the baseline: {speedup_str}"
    )


def build_multiturn_prompt(base_prompt: str, history: list[AttemptRecord]) -> str:
    """已弃用：仅保留供 log_prompt 等单轮拼接用；实际调用请用 build_feedback_block + messages 列表。"""
    if not history:
        return base_prompt
    return base_prompt.rstrip() + "\n\n" + build_feedback_block(history)


def build_feedback_block(history: list[AttemptRecord]) -> str:
    """仅构建「历史尝试与反馈」段落，作为新一轮的 user 消息内容（不含 base_prompt）。"""
    if not history:
        return ""
    block = "Here are your previous attempts:\n"
    for rec in history:
        block += "\n" + rec.kernel_code.strip() + "\n"
        if rec.summary:
            block += "\n" + rec.summary.strip() + "\n"
        block += "\n" + format_feedback_from_eval_result(rec.eval_result).strip() + "\n"
    block += "\nRestart your reasoning process and generate new, complete code.\n"
    return block


def _eval_with_retries(
    ref_arch_src: str,
    kernel_src: str,
    config: GenerationMultiturnAllLevelsConfig,
    max_retries: int = 3,
) -> KernelExecResult:
    """
    `eval_kernel_against_ref` returns None on some transient compilation lock errors.
    Retry a few times; if still None, convert to a parsing_failure-like result.
    """
    last_err: Optional[str] = None
    for i in range(max_retries):
        try:
            result = eval_kernel_against_ref(
                ref_arch_src,
                kernel_src,
                verbose=config.verbose,
                measure_performance=True,
                timing_method=getattr(config, "timing_method", "cuda_event"),
                backend=config.backend,
                precision=get_torch_dtype_from_string(config.precision),
            )
            if result is not None:
                return result
            last_err = "Eval returned None (likely compilation lock). Please retry."
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0 * (i + 1))
    return _make_parsing_failure_result(last_err or "Eval failed with unknown error.")


def generate_sample_multiturn_single(
    work: WorkArgs,
    config: GenerationMultiturnAllLevelsConfig,
    dataset,
    inference_server: callable,
    run_dir: str,
) -> bool:
    problem = dataset.get_problem_by_id(work.problem_id)
    ref_arch_src = problem.code
    problem_name = problem.name

    # Base prompt (shared across turns)
    if config.custom_prompt_key:
        base_prompt = get_custom_prompt(
            config.custom_prompt_key,
            ref_arch_src=ref_arch_src,
            backend=config.backend,
            option=config.prompt_option,
            precision=config.precision,
            include_hardware=config.include_hardware_info,
            gpu_name=config.hardware_gpu_name,
        )
    else:
        base_prompt = get_prompt_for_backend(
            ref_arch_src,
            config.backend,
            option=config.prompt_option,
            precision=config.precision,
            include_hardware=config.include_hardware_info,
            gpu_name=config.hardware_gpu_name,
        )

    history: list[AttemptRecord] = []
    last_kernel: Optional[str] = None
    start_turn = 0

    # 多轮对话记录；resume 时从 conversation.json 恢复
    conversation_messages: list[dict[str, str]] = []
    conv_path = os.path.join(
        run_dir,
        f"level_{config.level_label}_problem_{work.problem_id}_sample_{work.sample_id}_conversation.json",
    )
    if getattr(config, "log_conversation", True):
        system_prompt = getattr(config, "conversation_system_prompt", "") or ""
        if os.path.exists(conv_path):
            try:
                with open(conv_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                prev_messages = loaded.get("messages") or []
                prev_metrics = loaded.get("turn_metrics") or []
                if prev_messages and prev_metrics:
                    conversation_messages = prev_messages
                    start_turn = len(prev_metrics)
                    for i in range(start_turn):
                        turn_path = os.path.join(
                            run_dir,
                            f"level_{config.level_label}_problem_{work.problem_id}_sample_{work.sample_id}_turn_{i}_kernel.py",
                        )
                        if os.path.exists(turn_path):
                            with open(turn_path, "r", encoding="utf-8") as f:
                                kcode = f.read()
                        else:
                            kcode = ""
                        history.append(
                            AttemptRecord(
                                kernel_code=kcode,
                                summary=None,
                                eval_result=_turn_metric_to_eval_result(prev_metrics[i]),
                            )
                        )
                    if start_turn > 0:
                        last_kernel = history[-1].kernel_code or last_kernel
                    if config.verbose:
                        print(
                            f"[MultiTurn] Resuming level={config.level_label} problem={work.problem_id} sample={work.sample_id} from turn {start_turn}"
                        )
            except Exception as e:
                if config.verbose:
                    print(f"[MultiTurn] Could not load resume state from {conv_path}: {e}")
        if not conversation_messages:
            conversation_messages.append({"role": "system", "content": system_prompt})

    # 首轮任务末尾补充格式说明，便于模型按 code block 回复
    _first_turn_suffix = (
        "\n\nPlease reply with your complete code in a markdown code block (e.g. ```python ... ```). "
        "You may add a brief summary of changes after the code block."
    )

    for turn in range(start_turn, int(config.max_turns)):
        # 本轮 user 消息：首轮为完整任务描述，后续轮为「历史尝试与反馈」段落
        if turn == 0:
            user_content = base_prompt.rstrip() + _first_turn_suffix
        else:
            user_content = build_feedback_block(history)

        # 多轮对话：将已有 messages 与本轮 user 拼接后发送
        messages_to_send = conversation_messages + [{"role": "user", "content": user_content}]

        if config.log_prompt:
            prompt_path = os.path.join(
                run_dir,
                f"level_{config.level_label}_problem_{work.problem_id}_sample_{work.sample_id}_turn_{turn}_prompt.txt",
            )
            with open(prompt_path, "w") as f:
                f.write(user_content)

        raw = inference_server(messages_to_send)
        raw_str = (raw or "").strip()

        # 将本轮 user 与 assistant 追加到 conversation_messages，供下一轮拼接
        if getattr(config, "log_conversation", True):
            conversation_messages.append({"role": "user", "content": user_content})
            conversation_messages.append({"role": "assistant", "content": raw_str})

        kernel = extract_first_code(raw, ["python", "cpp"])
        summary = _extract_summary_after_first_codeblock(raw)

        if not kernel:
            history.append(
                AttemptRecord(
                    kernel_code=(raw or "").strip() or "<empty response>",
                    summary=summary,
                    eval_result=_make_parsing_failure_result("No fenced code block found in response."),
                )
            )
            if getattr(config, "log_conversation", True):
                _write_conversation_json(
                    run_dir, config.level_label, work.problem_id, work.sample_id, conversation_messages, history
                )
            continue

        # 每轮保存生成的 .py 脚本，便于按轮次查看
        turn_kernel_path = os.path.join(
            run_dir,
            f"level_{config.level_label}_problem_{work.problem_id}_sample_{work.sample_id}_turn_{turn}_kernel.py",
        )
        with open(turn_kernel_path, "w") as f:
            f.write(kernel)

        # Optional static checker; turn failures into feedback instead of crashing the whole job.
        if config.check_kernel:
            ok, error, warnings = validate_kernel_static(
                kernel,
                backend=config.backend,
                precision=config.precision,
            )
            if not ok:
                msg = f"Static check failed. Error: {error}. Warnings: {warnings}"
                history.append(
                    AttemptRecord(
                        kernel_code=kernel,
                        summary=summary,
                        eval_result=_make_parsing_failure_result(msg),
                    )
                )
                if getattr(config, "log_conversation", True):
                    _write_conversation_json(
                        run_dir, config.level_label, work.problem_id, work.sample_id, conversation_messages, history
                    )
                continue

        last_kernel = kernel

        try:
            eval_result = _eval_with_retries(ref_arch_src, kernel, config=config)
        except Exception as e:
            # CUDA 非法访问等异常可能从 eval 传出；捕获后记作本轮失败，继续下一轮
            if config.verbose:
                print(
                    f"[MultiTurn] level={config.level_label} problem={work.problem_id} sample={work.sample_id} "
                    f"turn={turn} eval exception: {e}"
                )
            eval_result = _make_parsing_failure_result(
                f"Eval raised exception: {type(e).__name__}: {e}"
            )

        if config.verbose:
            print(
                f"[MultiTurn] level={config.level_label} problem={work.problem_id} sample={work.sample_id} "
                f"turn={turn} name={problem_name} compiled={eval_result.compiled} correct={eval_result.correctness}"
            )

        history.append(
            AttemptRecord(
                kernel_code=kernel,
                summary=summary,
                eval_result=eval_result,
            )
        )

        if getattr(config, "log_conversation", True):
            _write_conversation_json(
                run_dir, config.level_label, work.problem_id, work.sample_id, conversation_messages, history
            )

        if config.early_stop_on_correct and eval_result.compiled and eval_result.correctness:
            break

    if last_kernel is None:
        raise RuntimeError(
            f"All turns failed to produce parseable code for problem {work.problem_id}: {problem_name}"
        )

    # Store final kernel (last produced in the multi-turn loop)
    kernel_path = os.path.join(
        run_dir,
        f"level_{config.level_label}_problem_{work.problem_id}_sample_{work.sample_id}_kernel.py",
    )
    with open(kernel_path, "w") as f:
        f.write(last_kernel)

    # 对话与 turn_metrics 已在每轮结束时写入，此处无需再写
    return True


def generate_sample_multiturn_launcher(
    work: WorkArgs,
    config: GenerationMultiturnAllLevelsConfig,
    dataset,
    inference_server: callable,
    run_dir: str,
):
    try:
        return generate_sample_multiturn_single(work, config, dataset, inference_server, run_dir)
    except Exception as e:
        print(f"Error generating multi-turn sample level={getattr(config,'level_label',None)} problem={work.problem_id} sample={work.sample_id}: {e}")
        return None


@pydra.main(base=GenerationMultiturnAllLevelsConfig)
def main(config: GenerationMultiturnAllLevelsConfig):
    """
    一次生成所有 level 的算子（Kevin 风格多轮 refine），无需指定 level 参数。
    """
    from kernelbench.utils import SERVER_PRESETS

    level_raw = getattr(config, "level", "all")
    level_str = str(level_raw).strip().lower()
    single_level: Optional[int] = None  # 若为 1/2/3/4 则只跑该 level，实现「只测一个问题」时配合 subset 使用
    if level_str in ("1", "2", "3", "4"):
        single_level = int(level_str)
    elif level_str not in ("all", ""):
        raise ValueError(
            "level 仅支持 all 或 1/2/3/4。指定 1/2/3/4 时只跑该 level（配合 subset 可只测一道题）。"
        )

    include_level4_expand = getattr(config, "include_level4_expand", False)
    if isinstance(include_level4_expand, str):
        include_level4_expand = include_level4_expand.lower() in ("true", "1", "yes")

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
        custom_prompt_key = None if trimmed.lower() in {"", "none"} else trimmed
    config.custom_prompt_key = custom_prompt_key

    include_hardware = config.include_hardware_info
    if isinstance(include_hardware, str):
        include_hardware = include_hardware.lower() in ["true", "1", "yes"]
    config.include_hardware_info = include_hardware

    supported_backends = {"cuda", "triton", "cute", "tilelang", "thunderkittens"}
    backend = config.backend.lower()
    if backend not in supported_backends:
        raise ValueError(
            f"Unsupported backend: {config.backend}. Must be one of {sorted(supported_backends)}."
        )
    config.backend = backend
    if backend == "tilelang":
        config.precision = "fp16"
    if backend == "thunderkittens":
        config.precision = "bf16"

    config.prompt_option = str(config.prompt_option).lower()
    valid_prompt_options = {"zero_shot", "one_shot", "few_shot"}
    if not config.custom_prompt_key:
        if config.prompt_option not in valid_prompt_options:
            raise ValueError(
                f"Invalid prompt_option '{config.prompt_option}'. Must be one of {sorted(valid_prompt_options)}."
            )
        if config.include_hardware_info and not config.hardware_gpu_name:
            raise ValueError(
                "include_hardware_info is True but hardware_gpu_name is not provided."
            )

    level_specs = get_all_level_specs(config.dataset_src, include_level4_expand)
    if single_level is not None:
        level_specs = [(s[0], s[1], s[2]) for s in level_specs if s[0] == single_level]
        if not level_specs:
            raise ValueError(f"level={single_level} 不在当前 level_specs 中。")
        print(f"Single-level mode: only level {single_level} (subset 生效时仅跑该 level 内题目).")
    print(f"Starting Multi-turn Batch Generation with config: {config}")
    print(f"Levels to generate: {[s[2] for s in level_specs]}")

    # 所有请求经 query_server，Host/api_base 由 .env 的 HOSTED_VLLM_*、LOCAL_SERVER_HOST 控制
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

    run_dir = os.path.join(config.runs_dir, config.run_name)
    run_exists = os.path.exists(run_dir)
    if run_exists and _run_dir_has_existing_kernels(run_dir):
        if not _ask_resume(run_dir):
            print(
                "   To start a fresh run, use a different run_name (e.g. run_name=my_run_v2). Exiting.\n"
            )
            sys.exit(0)
        print("   Resuming: will skip already generated kernels.\n")
    elif run_exists:
        print(f"\n⚠️  Run directory already exists: {run_dir}")
        print("   Existing kernels will be skipped. Use a different run_name for a fresh run.\n")
    os.makedirs(run_dir, exist_ok=True)

    save_dict = config.to_dict()
    save_dict["level"] = single_level if single_level is not None else "all"
    save_dict["include_level4_expand"] = include_level4_expand
    pydra.save_yaml(save_dict, os.path.join(run_dir, "generation_config.yaml"))

    assert config.store_type == "local", "Only local storage is supported."

    total_generated = 0
    total_attempted = 0
    total_failed = 0
    total_skipped = 0

    for dataset_level, local_subdir, level_label in level_specs:
        config.level_label = level_label
        dataset = construct_kernelbench_dataset(
            level=dataset_level,
            source=config.dataset_src,
            dataset_name=config.dataset_name,
            local_subdir=local_subdir,
        )
        all_problem_ids = dataset.get_problem_ids()

        if config.subset == (None, None):
            problem_ids_to_run = all_problem_ids
        else:
            start, end = config.subset
            problem_ids_to_run = [p for p in all_problem_ids if start <= p <= end]
            if not problem_ids_to_run:
                print(
                    f"Warning: No problems in subset {config.subset} for level {level_label}, skipping."
                )
                continue

        problems_to_run: list[WorkArgs] = []
        level_total = 0
        level_skipped = 0
        for problem_id in problem_ids_to_run:
            for sample_id in range(config.num_samples):
                level_total += 1
                if check_kernel_exists(run_dir, level_label, problem_id, sample_id):
                    level_skipped += 1
                else:
                    problems_to_run.append(WorkArgs(problem_id=int(problem_id), sample_id=sample_id))

        if level_skipped > 0:
            print(f"📁 Level {level_label}: {level_skipped}/{level_total} kernels already exist.")
        if not problems_to_run:
            print(f"Level {level_label}: no new kernels to generate.")
            total_skipped += level_total
            continue

        print(f"Level {level_label}: generating {len(problems_to_run)} kernels (of {level_total} total).")
        results = maybe_multithread(
            generate_sample_multiturn_launcher,
            problems_to_run,
            config.num_workers,
            time_interval=config.api_query_interval,
            config=config,
            dataset=dataset,
            inference_server=inference_server,
            run_dir=run_dir,
        )
        num_ok = len(results)
        num_fail = len(problems_to_run) - num_ok
        total_generated += num_ok
        total_attempted += len(problems_to_run)
        total_failed += num_fail
        total_skipped += level_skipped
        print(f"Level {level_label}: generated {num_ok}, failed {num_fail}, skipped {level_skipped}.")

    print("\n" + "=" * 60)
    print(
        f"All levels: generated {total_generated}, attempted {total_attempted}, failed {total_failed}, skipped {total_skipped}."
    )
    if total_attempted == 0 and total_skipped > 0:
        print(f"✅ All kernels already exist in {run_dir}")
    elif total_failed > 0:
        print(f"Please retry for the {total_failed} failed problems.")


if __name__ == "__main__":
    main()

