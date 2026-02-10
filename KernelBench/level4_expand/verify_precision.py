"""Verify numerical precision between official HuggingFace models and pure PyTorch implementations.

Uses RANDOM INITIALIZATION only (no from_pretrained/weight download) to validate:
1. state_dict key/order/shape compatibility with official Transformers
2. Numerical precision alignment (logits, hidden states) with identical random inputs
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
NUM_TRIALS = 3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _get_hf_model_class(model_type):
    """Get the appropriate HF model class for random init. Uses AutoModelForCausalLM.from_config for all."""
    from transformers import AutoModelForCausalLM
    return (AutoModelForCausalLM, None)


def _build_config_from_pretrained(model_name, model_type):
    """Build config from pretrained (config only, no weights) for random init."""
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_name)
    hf_config.tie_word_embeddings = False
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


def _compare_state_dicts(hf_sd, ref_sd, ref_prefix="model."):
    """Compare state_dict keys and shapes. Return (match, mismatches)."""
    hf_keys = sorted(hf_sd.keys())
    ref_keys = sorted(ref_sd.keys())

    mismatches = []
    # Refactored has "model." prefix
    ref_keys_bare = [k.replace(ref_prefix, "") if k.startswith(ref_prefix) else k for k in ref_keys]
    hf_set = set(hf_keys)
    ref_set = set(ref_keys_bare)

    if hf_set != ref_set:
        for k in hf_set - ref_set:
            mismatches.append(("missing_in_ref", k, None))
        for k in ref_set - hf_set:
            mismatches.append(("extra_in_ref", k, None))

    for hf_key in hf_keys:
        ref_key = ref_prefix + hf_key if any(
            k == ref_prefix + hf_key for k in ref_keys
        ) else hf_key
        if ref_key not in ref_sd:
            ref_key = hf_key
        if ref_key in ref_sd:
            if hf_sd[hf_key].shape != ref_sd[ref_key].shape:
                mismatches.append(
                    ("shape_mismatch", hf_key, (hf_sd[hf_key].shape, ref_sd[ref_key].shape))
                )

    return len(mismatches) == 0, mismatches


def verify_file(file_num, model_name, batch_size, sequence_length, model_type):
    """Verify precision for a single benchmark file using random init."""
    print(f"\n{'='*60}")
    print(f"File {file_num}: {model_name} (bs={batch_size}, seq={sequence_length})")
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
        hf_config = _build_config_from_pretrained(model_name, model_type)

        # Create OFFICIAL HF model with random init (no from_pretrained)
        from transformers import AutoModelForCausalLM
        original_model = AutoModelForCausalLM.from_config(hf_config)
        original_model.eval()

        # Create refactored model
        RefactoredModel = refactored_module.Model
        refactored_model = RefactoredModel(hf_config)
        refactored_model.eval()

        # Compare state_dict structure
        hf_sd = original_model.state_dict()
        ref_sd = refactored_model.state_dict()
        sd_match, sd_mismatches = _compare_state_dicts(hf_sd, ref_sd)

        if not sd_match:
            print("  [WARN] state_dict structure mismatch:")
            for m in sd_mismatches[:10]:
                print(f"    {m}")
            if len(sd_mismatches) > 10:
                print(f"    ... and {len(sd_mismatches) - 10} more")
        else:
            print("  [OK] state_dict keys/shapes compatible")

        # Copy weights from HF to refactored for precision test (in-place, avoid extra copy)
        # Refactored Model wraps in .model, so ref keys have "model." prefix
        for hf_key, hf_val in hf_sd.items():
            ref_key = "model." + hf_key
            if ref_key in ref_sd and ref_sd[ref_key].shape == hf_val.shape:
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
                original_out = original_model(inputs)
                original_logits = (
                    original_out.logits
                    if hasattr(original_out, "logits")
                    else original_out[0]
                )
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
        passed = worst_error <= THRESHOLD and sd_match

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
        description="Verify precision of refactored models (random init, no download)"
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
    args = parser.parse_args()

    threshold = args.threshold
    files_to_verify = args.files or [f[0] for f in FILES]

    print("Precision Verification Report (Random Init - No Weight Download)")
    print(f"Threshold: {threshold}")
    print(f"Files: {files_to_verify}")
    print(f"Device: {DEVICE}")

    results = []
    for entry in FILES:
        file_num, model_name, bs, seq = entry[0], entry[1], entry[2], entry[3]
        model_type = entry[4] if len(entry) > 4 else "auto"
        if file_num not in files_to_verify:
            continue
        result = verify_file(file_num, model_name, bs, seq, model_type)
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
