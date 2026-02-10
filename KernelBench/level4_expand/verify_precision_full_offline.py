"""Full offline precision verification - no network. Uses module configs for all models."""

import importlib
import os
import sys
import traceback

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# (file_num, module_name, batch_size, sequence_length)
FILES = [
    (1, "1_EleutherAI-gpt-neo-2p7B_bs32_seq256", 32, 256),
    (2, "2_facebook-opt-1p3b_bs1_seq2047", 1, 2047),
    (3, "3_EleutherAI-gpt-neo-2p7B_bs1_seq2047", 1, 2047),
    (4, "4_facebook-opt-1p3b_bs32_seq256", 32, 256),
    (5, "5_google-bigbird-roberta-base_bs1_seq4095", 1, 4095),
    (6, "6_facebook-bart-large_bs1_seq1023", 1, 1023),
    (7, "7_gpt2_bs32_seq256", 32, 256),
    (8, "8_facebook-opt-1p3b_bs512_seq32", 512, 32),
    (9, "9_google-bigbird-roberta-base_bs32_seq256", 32, 256),
    (10, "10_google-bigbird-roberta-base_bs1024_seq32", 1024, 32),
    (11, "11_google-electra-small-discriminator_bs1_seq511", 1, 511),
    (12, "12_google-electra-small-discriminator_bs1024_seq32", 1024, 32),
    (13, "13_google-reformer-enwik8_bs32_seq256", 32, 256),
    (14, "14_google-electra-small-discriminator_bs32_seq256", 32, 256),
    (15, "15_google-reformer-enwik8_bs1024_seq32", 1024, 32),
    (16, "16_gpt2_bs1_seq1023", 1, 1023),
    (17, "17_facebook-bart-large_bs1024_seq32", 1024, 32),
    (18, "18_EleutherAI-gpt-neo-2p7B_bs512_seq32", 512, 32),
    (19, "19_gpt2_bs1024_seq32", 1024, 32),
    (20, "20_facebook-bart-large_bs32_seq256", 32, 256),
]
THRESHOLD = 1e-5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_ref_module(module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(SCRIPT_DIR, f"{module_name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_gpt2(file_num, module_name, bs, seq_len):
    from transformers import GPT2Config, GPT2LMHeadModel
    mod = load_ref_module(module_name)
    cfg = mod.config
    hf_config = GPT2Config(
        vocab_size=cfg.vocab_size, n_positions=cfg.n_positions, n_embd=cfg.n_embd,
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_inner=cfg.n_inner,
        resid_pdrop=cfg.resid_pdrop, embd_pdrop=cfg.embd_pdrop, attn_pdrop=cfg.attn_pdrop,
        layer_norm_epsilon=cfg.layer_norm_epsilon,
        scale_attn_weights=cfg.scale_attn_weights,
        scale_attn_by_inverse_layer_idx=cfg.scale_attn_by_inverse_layer_idx,
    )
    hf_model = GPT2LMHeadModel._from_config(hf_config)
    ref_model = mod.Model(hf_config)
    hf_prefix, ref_prefix = "", "model."
    return _run_verify(file_num, module_name, bs, seq_len, hf_model, ref_model, mod, hf_prefix, ref_prefix)


def verify_gpt_neo(file_num, module_name, bs, seq_len):
    from transformers import GPTNeoConfig, GPTNeoForCausalLM
    mod = load_ref_module(module_name)
    cfg = getattr(mod, "config", None) or mod.get_init_inputs()[0]
    # GPTNeoConfig uses attention_types, not attention_layers; expand to num_layers
    attn_layers = getattr(cfg, "attention_layers", ["global"] * cfg.num_layers)
    attn_types = [[list(set(attn_layers[:1])), cfg.num_layers]] if attn_layers else [[["global"], cfg.num_layers]]
    hf_config = GPTNeoConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
        num_heads=cfg.num_heads, max_position_embeddings=cfg.max_position_embeddings,
        attention_types=[[["global"], cfg.num_layers]], window_size=cfg.window_size,
        embed_dropout=cfg.embed_dropout, attention_dropout=cfg.attention_dropout,
        resid_dropout=cfg.resid_dropout, layer_norm_epsilon=getattr(cfg, "layer_norm_epsilon", 1e-5),
    )
    hf_model = GPTNeoForCausalLM._from_config(hf_config)
    ref_model = mod.Model(hf_config)
    return _run_verify(file_num, module_name, bs, seq_len, hf_model, ref_model, mod, "", "model.")


def verify_opt(file_num, module_name, bs, seq_len):
    from transformers import OPTConfig, OPTForCausalLM
    mod = load_ref_module(module_name)
    cfg = getattr(mod, "config", None) or mod.get_init_inputs()[0]
    hf_config = OPTConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers, ffn_dim=cfg.ffn_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        num_attention_heads=cfg.num_attention_heads,
        word_embed_proj_dim=cfg.word_embed_proj_dim,
        dropout=cfg.dropout, activation_function=cfg.activation_function,
        do_layer_norm_before=cfg.do_layer_norm_before,
        enable_bias=cfg.enable_bias,
        layer_norm_elementwise_affine=cfg.layer_norm_elementwise_affine,
        pad_token_id=cfg.pad_token_id,
    )
    hf_model = OPTForCausalLM._from_config(hf_config)
    ref_model = mod.Model(hf_config)
    return _run_verify(file_num, module_name, bs, seq_len, hf_model, ref_model, mod, "model.", "model.model.")


def verify_bigbird(file_num, module_name, bs, seq_len):
    from transformers import BigBirdConfig, BigBirdForCausalLM
    mod = load_ref_module(module_name)
    cfg = getattr(mod, "config", None) or mod.get_init_inputs()[0]
    hf_config = BigBirdConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers, num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size, max_position_embeddings=cfg.max_position_embeddings,
        type_vocab_size=cfg.type_vocab_size, layer_norm_eps=cfg.layer_norm_eps,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
        pad_token_id=cfg.pad_token_id, attention_type=cfg.attention_type,
        block_size=cfg.block_size, num_random_blocks=cfg.num_random_blocks,
        rescale_embeddings=cfg.rescale_embeddings, use_bias=getattr(cfg, "use_bias", True),
        is_decoder=True,
    )
    hf_model = BigBirdForCausalLM._from_config(hf_config)
    ref_model = mod.Model(hf_config)
    return _run_verify(file_num, module_name, bs, seq_len, hf_model, ref_model, mod, "", "model.")


def verify_bart(file_num, module_name, bs, seq_len):
    from transformers import BartConfig, BartForCausalLM
    mod = load_ref_module(module_name)
    cfg = getattr(mod, "config", None) or mod.get_init_inputs()[0]
    hf_config = BartConfig(
        vocab_size=cfg.vocab_size, d_model=cfg.d_model,
        encoder_ffn_dim=cfg.encoder_ffn_dim, encoder_layers=cfg.encoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_ffn_dim=cfg.decoder_ffn_dim, decoder_layers=cfg.decoder_layers,
        decoder_attention_heads=cfg.decoder_attention_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        pad_token_id=cfg.pad_token_id, dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout, activation_dropout=cfg.activation_dropout,
        activation_function=cfg.activation_function,
    )
    hf_model = BartForCausalLM._from_config(hf_config)
    ref_model = mod.Model(hf_config)
    return _run_verify(file_num, module_name, bs, seq_len, hf_model, ref_model, mod, "model.", "model.model.")


def verify_electra(file_num, module_name, bs, seq_len):
    from transformers import ElectraConfig, ElectraForCausalLM
    mod = load_ref_module(module_name)
    cfg = getattr(mod, "config", None) or mod.ElectraConfig()
    hf_config = ElectraConfig(
        vocab_size=cfg.vocab_size, embedding_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        max_position_embeddings=cfg.max_position_embeddings,
        type_vocab_size=cfg.type_vocab_size, layer_norm_eps=cfg.layer_norm_eps,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
        pad_token_id=cfg.pad_token_id,
    )
    hf_model = ElectraForCausalLM._from_config(hf_config)
    ref_model = mod.Model(hf_config)
    return _run_verify(file_num, module_name, bs, seq_len, hf_model, ref_model, mod, "", "model.")


def verify_reformer(file_num, module_name, bs, seq_len):
    try:
        from transformers import ReformerConfig, ReformerModelWithLMHead
    except ImportError:
        from transformers import ReformerConfig
        ReformerModelWithLMHead = None
    if ReformerModelWithLMHead is None:
        return {"file": file_num, "passed": False, "error": float("inf"), "sd_ok": False, "module": module_name, "error_msg": "ReformerForCausalLM not available"}
    mod = load_ref_module(module_name)
    cfg = getattr(mod, "config", None) or mod.get_init_inputs()[0]
    hf_config = ReformerConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers, num_attention_heads=cfg.num_attention_heads,
        attention_head_size=cfg.attention_head_size,
        intermediate_size=cfg.intermediate_size, max_position_embeddings=cfg.max_position_embeddings,
        axial_pos_embeddings=True, axial_pos_shape=cfg.axial_pos_shape,
        axial_pos_embds_dim=cfg.axial_pos_embds_dim,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
        lsh_attn_chunk_length=cfg.lsh_attn_chunk_length,
        lsh_attention_probs_dropout_prob=cfg.lsh_attention_probs_dropout_prob,
        num_hashes=cfg.num_hashes, num_buckets=cfg.num_buckets,
    )
    hf_model = ReformerModelWithLMHead._from_config(hf_config)
    ref_model = mod.Model(hf_config)
    return _run_verify(file_num, module_name, bs, seq_len, hf_model, ref_model, mod, "", "model.")


def _run_verify(file_num, module_name, bs, seq_len, hf_model, ref_model, mod, hf_prefix, ref_prefix):
    print(f"\n=== File {file_num}: {module_name} (bs={bs}, seq={seq_len}) ===")
    try:
        hf_model.eval()
        ref_model.eval()

        hf_sd = hf_model.state_dict()
        ref_sd = ref_model.state_dict()
        ref_keys_bare = {k.replace("model.", "").replace("model.model.", "") for k in ref_sd}
        for k in list(ref_keys_bare):
            if k.startswith("model."):
                ref_keys_bare.add(k[6:])
        hf_keys = set(hf_sd.keys())
        sd_ok = hf_keys == ref_keys_bare or all(
            (ref_prefix.strip(".") + "." + k if not k.startswith("model.") else k) in ref_sd or ("model." + k) in ref_sd
            for k in hf_keys
        )
        if not sd_ok:
            missing = hf_keys - ref_keys_bare
            extra = ref_keys_bare - hf_keys
            if missing:
                print(f"  [WARN] Missing in ref: {list(missing)[:5]}")
            if extra:
                print(f"  [WARN] Extra in ref: {list(extra)[:5]}")
        else:
            print("  [OK] state_dict keys compatible")

        ref_sd_new = ref_model.state_dict()
        copied = 0
        for k, v in hf_sd.items():
            for rk in ["model." + k, "model.model." + k, k]:
                if rk in ref_sd_new and ref_sd_new[rk].shape == v.shape:
                    ref_sd_new[rk] = v.clone()
                    copied += 1
                    break
        ref_model.load_state_dict(ref_sd_new, strict=False)

        device = torch.device(DEVICE)
        hf_model = hf_model.to(device)
        ref_model = ref_model.to(device)
        vocab_size = mod.vocab_size

        torch.manual_seed(42)
        inputs = torch.randint(0, vocab_size, (bs, seq_len), device=device)

        with torch.no_grad():
            hf_out = hf_model(inputs)
            hf_logits = hf_out.logits if hasattr(hf_out, "logits") else hf_out[0]
            ref_out = ref_model(inputs)
            ref_logits = ref_out if not isinstance(ref_out, tuple) else ref_out[0]

        err = torch.max(torch.abs(hf_logits.float() - ref_logits.float())).item()
        passed = err <= THRESHOLD and sd_ok
        print(f"  Max abs error: {err:.2e}  [{'PASS' if passed else 'FAIL'}]")
        return {"file": file_num, "passed": passed, "error": err, "sd_ok": sd_ok, "module": module_name}
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return {"file": file_num, "passed": False, "error": float("inf"), "sd_ok": False, "module": module_name, "error_msg": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, nargs="*", default=None)
    args = parser.parse_args()
    files_to_run = args.files or [f[0] for f in FILES]

    DISPATCH = {
        1: verify_gpt_neo, 3: verify_gpt_neo, 18: verify_gpt_neo,
        2: verify_opt, 4: verify_opt, 8: verify_opt,
        5: verify_bigbird, 9: verify_bigbird, 10: verify_bigbird,
        6: verify_bart, 17: verify_bart, 20: verify_bart,
        11: verify_electra, 12: verify_electra, 14: verify_electra,
        13: verify_reformer, 15: verify_reformer,
        7: verify_gpt2, 16: verify_gpt2, 19: verify_gpt2,
    }

    results = []
    for fn, mn, bs, seq in FILES:
        if fn not in files_to_run:
            continue
        verifier = DISPATCH.get(fn)
        if verifier is None:
            results.append({"file": fn, "passed": False, "error": float("inf"), "module": mn, "error_msg": "No verifier"})
            continue
        try:
            results.append(verifier(fn, mn, bs, seq))
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"file": fn, "passed": False, "error": float("inf"), "module": mn, "error_msg": str(e)})

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'File':>5} {'Module':<45} {'Error':>12} {'Status':>6}")
    print("-" * 70)
    passed = 0
    for r in results:
        s = "PASS" if r["passed"] else "FAIL"
        err = r.get("error", float("inf"))
        mod = r.get("module", "")
        print(f"{r['file']:>5} {mod[:44]:<45} {err:>12.2e} {s:>6}")
        if r["passed"]:
            passed += 1
    print("-" * 70)
    print(f"Total: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
