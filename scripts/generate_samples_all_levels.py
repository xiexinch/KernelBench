"""
Batch Generate Samples for All Levels (1, 2, 3, 4; optionally level4_expand).

One run generates kernels for all levels without specifying a level parameter.
Each script is independent; level list logic is defined locally.
"""

from typing import Optional

import os
import pydra

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.utils import maybe_multithread

from generate_samples import (
    GenerationConfig,
    WorkArgs,
    check_kernel_exists,
    generate_sample_launcher,
)


REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_all_level_specs(
    dataset_src: str, include_level4_expand: bool
) -> list[tuple[int, Optional[str], str]]:
    """
    返回要跑的所有 level 的 (dataset_level, local_subdir, level_label) 列表。
    level_label 用于文件命名，如 level_1、level_level4_expand。
    """
    specs = [
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


class GenerationAllLevelsConfig(GenerationConfig):
    """Config for generating all levels; level is optional and defaults to 'all'."""

    def __init__(self):
        super().__init__()
        self.level = "all"
        self.include_level4_expand = False


@pydra.main(base=GenerationAllLevelsConfig)
def main(config: GenerationAllLevelsConfig):
    """
    一次生成所有 level 的算子，无需指定 level 参数。
    """
    from kernelbench.utils import SERVER_PRESETS

    level_raw = getattr(config, "level", "all")
    if str(level_raw).strip().lower() not in ("all", ""):
        raise ValueError(
            "本脚本仅支持生成所有 level。若需单 level，请使用 generate_samples.py 并指定 level 参数。"
        )

    include_level4_expand = getattr(config, "include_level4_expand", False)
    if isinstance(include_level4_expand, str):
        include_level4_expand = include_level4_expand.lower() in ("true", "1", "yes")

    # 与 generate_samples.main 一致的 config 校验与预设解析（只做一次）
    if config.server_type and config.server_type in SERVER_PRESETS:
        preset = SERVER_PRESETS[config.server_type]
        if config.model_name is None or config.model_name == "None":
            config.model_name = preset.get("model_name", "None")
        if config.max_tokens is None or config.max_tokens == "None":
            config.max_tokens = preset.get("max_tokens", "None")
        if config.temperature is None or config.temperature == "None":
            config.temperature = preset.get("temperature", "None")

    if isinstance(config.is_reasoning_model, str):
        config.is_reasoning_model = config.is_reasoning_model.lower() in [
            "true",
            "1",
            "yes",
        ]

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
    print(f"Starting Batch Generation for all levels with config: {config}")
    print(f"Levels to generate: {[s[2] for s in level_specs]}")

    from kernelbench.utils import create_inference_server_from_presets

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
    if run_exists:
        print(f"\n⚠️  WARNING: Run directory already exists: {run_dir}")
        print(
            "   Existing kernels will be skipped. Use a different run_name for a fresh run.\n"
        )
    os.makedirs(run_dir, exist_ok=True)
    save_dict = config.to_dict()
    save_dict["level"] = "all"
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

        problems_to_run = []
        level_total = 0
        level_skipped = 0
        for problem_id in problem_ids_to_run:
            for sample_id in range(config.num_samples):
                level_total += 1
                if check_kernel_exists(run_dir, level_label, problem_id, sample_id):
                    level_skipped += 1
                else:
                    problems_to_run.append(
                        WorkArgs(problem_id=int(problem_id), sample_id=sample_id)
                    )

        if level_skipped > 0:
            print(
                f"📁 Level {level_label}: {level_skipped}/{level_total} kernels already exist."
            )
        if not problems_to_run:
            print(f"Level {level_label}: no new kernels to generate.")
            total_skipped += level_total
            continue

        print(
            f"Level {level_label}: generating {len(problems_to_run)} kernels (of {level_total} total)."
        )
        results = maybe_multithread(
            generate_sample_launcher,
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
        print(
            f"Level {level_label}: generated {num_ok}, failed {num_fail}, skipped {level_skipped}."
        )

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
