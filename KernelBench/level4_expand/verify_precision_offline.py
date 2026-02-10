"""Offline precision verification - no network, uses module configs.

Run: uv run python KernelBench/level4_expand/verify_precision_offline.py --files 7 16 19
"""

import importlib
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# (file_num, module_name, batch_size, sequence_length)
FILES = [
    (7, "7_gpt2_bs32_seq256", 32, 256),
    (16, "16_gpt2_bs1_seq1023", 1, 1023),
    (19, "19_gpt2_bs1024_seq32", 1024, 32),
]
THRESHOLD = 1e-5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def verify_gpt2(file_num, module_name, bs, seq_len):
    """Verify GPT2-level4_expand vs official GPT2LMHeadModel (random init)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    print(f"\n=== File {file_num}: {module_name} (bs={bs}, seq={seq_len}) ===")

    # Load refactored
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(SCRIPT_DIR, f"{module_name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Build config from refactored module's config
    cfg = mod.config
    hf_config = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.n_positions,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_inner=cfg.n_inner,
        resid_pdrop=cfg.resid_pdrop,
        embd_pdrop=cfg.embd_pdrop,
        attn_pdrop=cfg.attn_pdrop,
        layer_norm_epsilon=cfg.layer_norm_epsilon,
        scale_attn_weights=cfg.scale_attn_weights,
        scale_attn_by_inverse_layer_idx=cfg.scale_attn_by_inverse_layer_idx,
    )

    # Official HF model (random init)
    hf_model = GPT2LMHeadModel._from_config(hf_config)
    hf_model.eval()

    # Refactored
    ref_model = mod.Model(hf_config)
    ref_model.eval()

    # Compare state_dict
    hf_sd = hf_model.state_dict()
    ref_sd = ref_model.state_dict()
    ref_keys_bare = {k.replace("model.", "") for k in ref_sd if k.startswith("model.")}
    hf_keys = set(hf_sd.keys())
    sd_ok = hf_keys == ref_keys_bare
    if not sd_ok:
        missing = hf_keys - ref_keys_bare
        extra = ref_keys_bare - hf_keys
        if missing:
            print(f"  [WARN] Missing in ref: {list(missing)[:5]}")
        if extra:
            print(f"  [WARN] Extra in ref: {list(extra)[:5]}")
    else:
        print("  [OK] state_dict keys compatible")

    # Copy weights: ref uses "model." prefix
    ref_sd_new = ref_model.state_dict()
    for k, v in hf_sd.items():
        rk = "model." + k
        if rk in ref_sd_new and ref_sd_new[rk].shape == v.shape:
            ref_sd_new[rk] = v.clone()
    ref_model.load_state_dict(ref_sd_new, strict=False)

    # Precision test
    device = torch.device(DEVICE)
    hf_model = hf_model.to(device)
    ref_model = ref_model.to(device)
    vocab_size = mod.vocab_size

    torch.manual_seed(42)
    inputs = torch.randint(0, vocab_size, (bs, seq_len), device=device)

    with torch.no_grad():
        hf_logits = hf_model(inputs).logits
        ref_logits = ref_model(inputs)

    err = torch.max(torch.abs(hf_logits.float() - ref_logits.float())).item()
    passed = err <= THRESHOLD and sd_ok
    print(f"  Max abs error: {err:.2e}  [{'PASS' if passed else 'FAIL'}]")
    return {"file": file_num, "passed": passed, "error": err, "sd_ok": sd_ok}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, nargs="*", default=[7, 16, 19])
    args = parser.parse_args()

    results = []
    for fn, mn, bs, seq in FILES:
        if fn in args.files:
            results.append(verify_gpt2(fn, mn, bs, seq))

    print("\n=== SUMMARY ===")
    for r in results:
        s = "PASS" if r["passed"] else "FAIL"
        print(f"  File {r['file']}: error={r['error']:.2e} {s}")
    passed = sum(1 for r in results if r["passed"])
    print(f"Total: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
