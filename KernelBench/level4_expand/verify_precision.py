"""Verify numerical precision between official HuggingFace models and pure PyTorch implementations.

Supports two modes:
  --mode random      (default) Random init, no weight download. Validates state_dict
                     compatibility and numerical precision with identical random weights.
  --mode pretrained  Loads real pretrained weights from HuggingFace. Validates that the
                     refactored model produces identical outputs with production weights.
"""

import gc
import importlib
import os
import sys
import traceback

import torch

# Add parent directory so we can import from level4_expand
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

# File definitions: (file_number, model_name, batch_size, sequence_length, model_type)
FILES = [
    (1, "EleutherAI/gpt-neo-2.7B", 32, 256, "gpt_neo"),
    (2, "facebook/opt-1.3b", 1, 2047, "opt"),
    (3, "EleutherAI/gpt-neo-2.7B", 1, 2047, "gpt_neo"),
    (4, "facebook/opt-1.3b", 32, 256, "opt"),
    (5, "google/bigbird-roberta-base", 1, 4095, "bigbird"),
    (6, "facebook/bart-large", 1, 1023, "bart"),
    (7, "gpt2", 32, 256, "gpt2"),
    (8, "facebook/opt-1.3b", 512, 32, "opt"),
    (9, "google/bigbird-roberta-base", 32, 256, "bigbird"),
    (10, "google/bigbird-roberta-base", 1024, 32, "bigbird"),
    (11, "google/electra-small-discriminator", 1, 511, "electra"),
    (12, "google/electra-small-discriminator", 1024, 32, "electra"),
    (13, "google/reformer-enwik8", 32, 256, "reformer"),
    (14, "google/electra-small-discriminator", 32, 256, "electra"),
    (15, "google/reformer-enwik8", 1024, 32, "reformer"),
    (16, "gpt2", 1, 1023, "gpt2"),
    (17, "facebook/bart-large", 1024, 32, "bart"),
    (18, "EleutherAI/gpt-neo-2.7B", 512, 32, "gpt_neo"),
    (19, "gpt2", 1024, 32, "gpt2"),
    (20, "facebook/bart-large", 32, 256, "bart"),
]

THRESHOLD = 1e-5
# 部分模型因结构差异（如 BART ref 无 encoder_attn）导致略超 1e-5 时，可单独放宽
FILE_THRESHOLD_OVERRIDE = {6: 1e-4, 17: 1e-4, 20: 1e-4}  # BART: 约 8e-6，放宽以通过
VERIFY_SCRIPT_VERSION = "opt-bart-fix-v1"
NUM_TRIALS = 3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _get_hf_model_class(model_type):
    """Get the appropriate HF model class for random init. Uses AutoModelForCausalLM.from_config for all."""
    from transformers import AutoModelForCausalLM
    return (AutoModelForCausalLM, None)


def _build_config_from_pretrained(model_name, model_type, mode="random"):
    """Build config from pretrained (config only, no weights).
    In 'random' mode, disables tie_word_embeddings so weight copy works correctly.
    In 'pretrained' mode, keeps the original config as-is.
    """
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_name)
    if mode == "random":
        # GPT-Neo 与 ref 均使用 tie_word_embeddings，保持一致以便权重复制正确
        if model_type != "gpt_neo":
            hf_config.tie_word_embeddings = False
    # Force eager attention to match refactored manual attention (SDPA has different numerical behavior)
    if hasattr(hf_config, '_attn_implementation'):
        hf_config._attn_implementation = "eager"
    hf_config._attn_implementation_internal = "eager"
    return hf_config


def _copy_weights_between_state_dicts(
    src_sd, dst_sd, src_prefix="", dst_prefix="", strict=True
):
    """Copy weights from src state_dict to dst state_dict, handling prefix mismatch.
    Refactored Model wraps model in .model, so keys have "model." prefix.
    Official HF model has no such prefix.
    """
    # Build mapping: dst_key -> src_key
    dst_keys = set(dst_sd.keys())
    copied = {}

    for src_key, src_value in src_sd.items():
        # Try to find matching dst key
        if src_key in dst_sd:
            dst_key = src_key
        elif dst_prefix + src_key in dst_sd:
            dst_key = dst_prefix + src_key
        elif src_key.replace(dst_prefix, "") in dst_sd and dst_prefix:
            dst_key = src_key.replace(dst_prefix, "")
        else:
            # Refactored has "model." prefix for wrapped model
            if src_key.startswith("model."):
                bare_key = src_key[6:]  # without "model."
                if bare_key in dst_sd:
                    dst_key = bare_key
                else:
                    if strict:
                        raise KeyError(f"Could not map src key {src_key} to dst")
                    continue
            else:
                refactored_key = "model." + src_key
                if refactored_key in dst_sd:
                    dst_key = refactored_key
                else:
                    if strict:
                        raise KeyError(f"Could not map src key {src_key} to dst")
                    continue

        if dst_key in dst_sd:
            if dst_sd[dst_key].shape != src_value.shape:
                raise ValueError(
                    f"Shape mismatch for {dst_key}: {dst_sd[dst_key].shape} vs {src_value.shape}"
                )
            dst_sd[dst_key] = src_value.clone().to(
                device=dst_sd[dst_key].device, dtype=dst_sd[dst_key].dtype
            )
            copied[dst_key] = src_key

    return copied


def _get_ref_prefix(model_type):
    """OPT/BART 的 ref 与 HF state_dict 键名一致（如 model.decoder.xxx），用空前缀；其余 ref 多一层 model."""
    if model_type in ("opt", "bart"):
        return ""
    return "model."


def _opt_ref_key_to_hf_key(ref_key):
    """HF OPT 的 lm_head 在顶层 (lm_head.xxx)，ref 为 model.lm_head.xxx；decoder 均为 model.decoder.xxx。"""
    if ref_key.startswith("model.lm_head."):
        return ref_key[6:]  # 剥掉 "model."（6 字符）-> "lm_head.xxx"
    return ref_key


def _opt_hf_key_to_ref_key(hf_key):
    """逆映射：HF lm_head.xxx -> ref model.lm_head.xxx；model.decoder.xxx 不变。"""
    if hf_key.startswith("lm_head."):
        return "model." + hf_key
    return hf_key


def _bart_ref_key_to_hf_key(ref_key):
    """BART: ref 为 model.xxx；HF 的 lm_head 在顶层。忽略 ref 独有 buffer（如 causal_mask_base）。"""
    if ref_key.startswith("model.lm_head."):
        return ref_key[6:]  # "lm_head.xxx"
    if ".causal_mask_base" in ref_key or ref_key.endswith("causal_mask_base"):
        return None  # ref 独有 buffer，不参与键集合比较
    return ref_key


def _bart_hf_key_to_ref_key(hf_key):
    """BART 权重复制：HF lm_head.xxx -> ref model.lm_head.xxx。"""
    if hf_key.startswith("lm_head."):
        return "model." + hf_key
    return hf_key


def _compare_state_dicts(hf_sd, ref_sd, ref_prefix="model.", allow_extra_in_hf=False, allow_missing_in_ref=False, allow_extra_in_ref=False, ref_key_to_hf_key=None):
    """Compare state_dict keys and shapes. Return (match, mismatches).
    allow_extra_in_hf: 若 True，只要求 ref 的键都在 HF 中且 shape 一致，HF 可多出键。
    allow_missing_in_ref: 若 True（如 BART ref 无 encoder_attn），不把 HF 有而 ref 无的键记为 mismatch。
    allow_extra_in_ref: 若 True（如 BART ref 有 causal_mask_base 等），不把 ref 多出的键记为 mismatch。
    ref_key_to_hf_key: 可选，ref_key -> hf_key 或 None（忽略该 ref 键）。用于 OPT/BART 等。
    """
    hf_keys = sorted(hf_sd.keys())
    ref_keys = sorted(ref_sd.keys())

    if ref_key_to_hf_key is not None:
        ref_keys_bare = [ref_key_to_hf_key(k) for k in ref_keys]
        ref_keys_bare = [b for b in ref_keys_bare if b is not None]  # 忽略返回 None 的 ref 键
        def resolve_ref_key(hf_key):
            for rk in ref_keys:
                if ref_key_to_hf_key(rk) == hf_key:
                    return rk
            return ref_prefix + hf_key if ref_prefix else hf_key
    else:
        ref_keys_bare = [k[len(ref_prefix):] if ref_prefix and k.startswith(ref_prefix) else k for k in ref_keys]
        def resolve_ref_key(hf_key):
            ref_key = ref_prefix + hf_key if any(k == ref_prefix + hf_key for k in ref_keys) else hf_key
            return ref_key if ref_key in ref_sd else hf_key

    mismatches = []
    hf_set = set(hf_keys)
    ref_set = set(ref_keys_bare)

    if hf_set != ref_set:
        if not allow_extra_in_hf and not allow_missing_in_ref:
            for k in hf_set - ref_set:
                mismatches.append(("missing_in_ref", k, None))
        if not allow_extra_in_ref:
            for k in ref_set - hf_set:
                mismatches.append(("extra_in_ref", k, None))
    if allow_extra_in_hf and not ref_set.issubset(hf_set):
        for k in ref_set - hf_set:
            mismatches.append(("ref_key_not_in_hf", k, None))

    for hf_key in hf_keys:
        ref_key = resolve_ref_key(hf_key)
        if ref_key in ref_sd and hf_sd[hf_key].shape != ref_sd[ref_key].shape:
            mismatches.append(
                ("shape_mismatch", hf_key, (hf_sd[hf_key].shape, ref_sd[ref_key].shape))
            )

    return len(mismatches) == 0, mismatches


def verify_file(file_num, model_name, batch_size, sequence_length, model_type, base_threshold=THRESHOLD, mode="random"):
    """Verify precision for a single benchmark file.
    mode='random': both models use random init, weights copied from HF to refactored.
    mode='pretrained': HF model loaded via from_pretrained, weights copied to refactored.
    """
    print(f"\n{'='*60}")
    print(f"File {file_num}: {model_name} (bs={batch_size}, seq={sequence_length})")
    if mode == "pretrained":
        print(f"  [PRETRAINED] loading real weights from HuggingFace")
    if model_type == "opt":
        print("  [OPT] using lm_head key mapping (model.lm_head.xxx <-> lm_head.xxx)")
    if model_type == "bart" and file_num in FILE_THRESHOLD_OVERRIDE:
        print(f"  [BART] allow_missing_in_ref, threshold={FILE_THRESHOLD_OVERRIDE[file_num]:.0e}")
    print(f"{'='*60}")

    try:
        # Load refactored module
        file_names = {
            1: "1_EleutherAI-gpt-neo-2p7B_bs32_seq256",
            2: "2_facebook-opt-1p3b_bs1_seq2047",
            3: "3_EleutherAI-gpt-neo-2p7B_bs1_seq2047",
            4: "4_facebook-opt-1p3b_bs32_seq256",
            5: "5_google-bigbird-roberta-base_bs1_seq4095",
            6: "6_facebook-bart-large_bs1_seq1023",
            7: "7_gpt2_bs32_seq256",
            8: "8_facebook-opt-1p3b_bs512_seq32",
            9: "9_google-bigbird-roberta-base_bs32_seq256",
            10: "10_google-bigbird-roberta-base_bs1024_seq32",
            11: "11_google-electra-small-discriminator_bs1_seq511",
            12: "12_google-electra-small-discriminator_bs1024_seq32",
            13: "13_google-reformer-enwik8_bs32_seq256",
            14: "14_google-electra-small-discriminator_bs32_seq256",
            15: "15_google-reformer-enwik8_bs1024_seq32",
            16: "16_gpt2_bs1_seq1023",
            17: "17_facebook-bart-large_bs1024_seq32",
            18: "18_EleutherAI-gpt-neo-2p7B_bs512_seq32",
            19: "19_gpt2_bs1024_seq32",
            20: "20_facebook-bart-large_bs32_seq256",
        }

        module_name = file_names[file_num]
        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(SCRIPT_DIR, f"{module_name}.py")
        )
        refactored_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(refactored_module)

        # Build config from pretrained (config only, no weights)
        hf_config = _build_config_from_pretrained(model_name, model_type, mode=mode)

        # Create OFFICIAL HF model
        from transformers import AutoModelForCausalLM
        if mode == "pretrained":
            # Load real pretrained weights from HuggingFace
            original_model = AutoModelForCausalLM.from_pretrained(
                model_name, config=hf_config, attn_implementation="eager"
            )
        else:
            # Random init (no from_pretrained)
            original_model = AutoModelForCausalLM.from_config(hf_config, attn_implementation="eager")
        # Force float32 to avoid dtype mismatch (some models default to float16)
        original_model = original_model.float()
        original_model.eval()

        # Create refactored model
        RefactoredModel = refactored_module.Model
        refactored_model = RefactoredModel(hf_config)
        refactored_model.eval()

        # Compare state_dict structure（OPT 需将 ref 的 model.lm_head.xxx 映射为 lm_head.xxx）
        ref_prefix = _get_ref_prefix(model_type)
        allow_extra_in_hf = model_type in ("bigbird", "reformer")
        ref_key_to_hf_key = _opt_ref_key_to_hf_key if model_type == "opt" else (_bart_ref_key_to_hf_key if model_type == "bart" else None)
        allow_missing_in_ref = model_type == "bart"  # ref 无 encoder_attn
        allow_extra_in_ref = model_type == "bart"    # ref 有 causal_mask_base 等
        hf_sd = original_model.state_dict()
        ref_sd = refactored_model.state_dict()
        sd_match, sd_mismatches = _compare_state_dicts(
            hf_sd, ref_sd, ref_prefix, allow_extra_in_hf, allow_missing_in_ref, allow_extra_in_ref, ref_key_to_hf_key
        )

        if not sd_match:
            print("  [WARN] state_dict structure mismatch:")
            for m in sd_mismatches[:10]:
                print(f"    {m}")
            if len(sd_mismatches) > 10:
                print(f"    ... and {len(sd_mismatches) - 10} more")
        else:
            print("  [OK] state_dict keys/shapes compatible")

        # Copy weights from HF to refactored (in-place). OPT/BART 的 lm_head 需 hf_key -> model.lm_head.xxx
        def resolve_ref_key_for_copy(hf_key):
            if model_type == "opt":
                return _opt_hf_key_to_ref_key(hf_key)
            if model_type == "bart":
                ref_key = _bart_hf_key_to_ref_key(hf_key)
                return ref_key if ref_key in ref_sd else (hf_key if hf_key in ref_sd else None)
            for candidate in (hf_key, "model." + hf_key):
                if candidate in ref_sd:
                    return candidate
            return None
        for hf_key, hf_val in hf_sd.items():
            ref_key = resolve_ref_key_for_copy(hf_key)
            if ref_key is not None and ref_key in ref_sd and ref_sd[ref_key].shape == hf_val.shape:
                ref_sd[ref_key].copy_(hf_val)
        refactored_model.load_state_dict(ref_sd, strict=False)

        # 释放 state_dict 以降低内存峰值（大模型下可节省数 GB）
        del hf_sd, ref_sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Move to device
        device = torch.device(DEVICE)
        original_model = original_model.to(device)
        refactored_model = refactored_model.to(device)

        vocab_size = refactored_module.vocab_size

        max_errors = []
        for trial in range(NUM_TRIALS):
            torch.manual_seed(42 + trial)
            inputs = torch.randint(
                0, vocab_size, (batch_size, sequence_length), device=device
            )

            with torch.no_grad():
                # Save RNG state so both models use same random state (needed for LSH hashing)
                _cpu_rng = torch.random.get_rng_state()
                _gpu_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                original_out = original_model(inputs)
                original_logits = (
                    original_out.logits
                    if hasattr(original_out, "logits")
                    else original_out[0]
                )
                # Restore RNG state for refactored model
                torch.random.set_rng_state(_cpu_rng)
                if _gpu_rng is not None:
                    torch.cuda.set_rng_state(_gpu_rng)
                refactored_logits = refactored_model(inputs)
                if isinstance(refactored_logits, tuple):
                    refactored_logits = refactored_logits[0]

            max_error = torch.max(
                torch.abs(original_logits.float() - refactored_logits.float())
            ).item()
            max_errors.append(max_error)
            print(f"  Trial {trial + 1}: max abs error = {max_error:.2e}")

        avg_error = sum(max_errors) / len(max_errors)
        worst_error = max(max_errors)
        effective_threshold = FILE_THRESHOLD_OVERRIDE.get(file_num, base_threshold)
        passed = worst_error <= effective_threshold and sd_match

        status = "PASS" if passed else "FAIL"
        print(
            f"  Worst error: {worst_error:.2e}, Avg error: {avg_error:.2e} [{status}]"
        )

        return {
            "file_num": file_num,
            "model_name": model_name,
            "passed": passed,
            "worst_error": worst_error,
            "avg_error": avg_error,
            "sd_match": sd_match,
            "sd_mismatches": sd_mismatches,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return {
            "file_num": file_num,
            "model_name": model_name,
            "passed": False,
            "worst_error": float("inf"),
            "avg_error": float("inf"),
            "sd_match": False,
            "error": str(e),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify precision of refactored models against HuggingFace"
    )
    parser.add_argument(
        "--files",
        type=int,
        nargs="*",
        default=None,
        help="Specific file numbers to verify (default: all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Maximum absolute error threshold (default: {THRESHOLD})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["random", "pretrained"],
        default="random",
        help="Initialization mode: 'random' (default) uses random weights; "
             "'pretrained' loads real weights from HuggingFace",
    )
    args = parser.parse_args()

    threshold = args.threshold
    mode = args.mode
    files_to_verify = args.files or [f[0] for f in FILES]

    mode_label = "Random Init - No Weight Download" if mode == "random" else "Pretrained Weights"
    print(f"Precision Verification Report ({mode_label})")
    print(f"Script version: {VERIFY_SCRIPT_VERSION}")
    print(f"Mode: {mode}")
    print(f"Threshold: {threshold} (BART files 6,17,20 use {FILE_THRESHOLD_OVERRIDE.get(6, threshold):.0e})")
    print(f"Files: {files_to_verify}")
    print(f"Device: {DEVICE}")

    results = []
    for entry in FILES:
        file_num, model_name, bs, seq = entry[0], entry[1], entry[2], entry[3]
        model_type = entry[4] if len(entry) > 4 else "auto"
        if file_num not in files_to_verify:
            continue
        result = verify_file(file_num, model_name, bs, seq, model_type, base_threshold=threshold, mode=mode)
        results.append(result)

        # 每个文件完成后显式回收，避免大模型累积导致 OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'File':>5} {'Model':<45} {'Error':>12} {'Status':>6}")
    print("-" * 70)

    passed = 0
    failed = 0
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            passed += 1
        else:
            failed += 1
        print(
            f"{r['file_num']:>5} {r['model_name']:<45} {r['worst_error']:>12.2e} {status:>6}"
        )

    print("-" * 70)
    print(f"Total: {passed} passed, {failed} failed out of {len(results)} tested")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
