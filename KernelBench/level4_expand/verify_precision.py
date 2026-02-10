"""Verify numerical precision between original HuggingFace models and pure PyTorch implementations."""

import importlib
import os
import sys
import traceback

import torch

# Add parent directory so we can import from level4 (original) and level4_expand (refactored)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

# File definitions: (file_number, model_name, batch_size, sequence_length)
FILES = [
    (1, "EleutherAI/gpt-neo-2.7B", 32, 256),
    (2, "facebook/opt-1.3b", 1, 2047),
    (3, "EleutherAI/gpt-neo-2.7B", 1, 2047),
    (4, "facebook/opt-1.3b", 32, 256),
    (5, "google/bigbird-roberta-base", 1, 4095),
    (6, "facebook/bart-large", 1, 1023),
    (7, "gpt2", 32, 256),
    (8, "facebook/opt-1.3b", 512, 32),
    (9, "google/bigbird-roberta-base", 32, 256),
    (10, "google/bigbird-roberta-base", 1024, 32),
    (11, "google/electra-small-discriminator", 1, 511),
    (12, "google/electra-small-discriminator", 1024, 32),
    (13, "google/reformer-enwik8", 32, 256),
    (14, "google/electra-small-discriminator", 32, 256),
    (15, "google/reformer-enwik8", 1024, 32),
    (16, "gpt2", 1, 1023),
    (17, "facebook/bart-large", 1024, 32),
    (18, "EleutherAI/gpt-neo-2.7B", 512, 32),
    (19, "gpt2", 1024, 32),
    (20, "facebook/bart-large", 32, 256),
]

THRESHOLD = 1e-5
NUM_TRIALS = 3
DEVICE = "cpu"


def load_original_model(model_name, config):
    """Load original HuggingFace model."""
    from transformers import AutoModelForCausalLM, AutoConfig
    hf_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=hf_config)
    model.eval()
    return model


def verify_file(file_num, model_name, batch_size, sequence_length):
    """Verify precision for a single benchmark file."""
    print(f"\n{'='*60}")
    print(f"File {file_num}: {model_name} (bs={batch_size}, seq={sequence_length})")
    print(f"{'='*60}")

    try:
        # Load refactored model
        # Import the benchmark file dynamically
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

        # Get init inputs and create refactored model
        init_inputs = refactored_module.get_init_inputs()
        RefactoredModel = refactored_module.Model
        refactored_model = RefactoredModel(*init_inputs)
        refactored_model.eval()

        # Load original model
        from transformers import AutoModelForCausalLM, AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name)
        original_model = AutoModelForCausalLM.from_pretrained(model_name, config=hf_config)
        original_model.eval()

        vocab_size = refactored_module.vocab_size

        max_errors = []
        for trial in range(NUM_TRIALS):
            torch.manual_seed(42 + trial)
            inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))

            with torch.no_grad():
                original_logits = original_model(inputs).logits
                refactored_logits = refactored_model(inputs)

            max_error = torch.max(torch.abs(original_logits - refactored_logits)).item()
            max_errors.append(max_error)
            print(f"  Trial {trial + 1}: max abs error = {max_error:.2e}")

        avg_error = sum(max_errors) / len(max_errors)
        worst_error = max(max_errors)
        passed = worst_error <= THRESHOLD

        status = "PASS" if passed else "FAIL"
        print(f"  Worst error: {worst_error:.2e}, Avg error: {avg_error:.2e} [{status}]")

        return {
            "file_num": file_num,
            "model_name": model_name,
            "passed": passed,
            "worst_error": worst_error,
            "avg_error": avg_error,
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
            "error": str(e),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify precision of refactored models")
    parser.add_argument("--files", type=int, nargs="*", default=None,
                        help="Specific file numbers to verify (default: all)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Maximum absolute error threshold (default: {THRESHOLD})")
    args = parser.parse_args()

    threshold = args.threshold
    files_to_verify = args.files or [f[0] for f in FILES]

    print(f"Precision Verification Report")
    print(f"Threshold: {threshold}")
    print(f"Files: {files_to_verify}")

    results = []
    for file_num, model_name, bs, seq in FILES:
        if file_num not in files_to_verify:
            continue
        result = verify_file(file_num, model_name, bs, seq)
        results.append(result)

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
        print(f"{r['file_num']:>5} {r['model_name']:<45} {r['worst_error']:>12.2e} {status:>6}")

    print("-" * 70)
    print(f"Total: {passed} passed, {failed} failed out of {len(results)} tested")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
