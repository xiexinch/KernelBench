"""Self-contained GPT-2 benchmark (batch_size=1, sequence_length=1023).

This file embeds all necessary code for config loading, weight downloading,
and the GPT-2 model implementation to avoid dependencies on the models folder.
"""

import json
import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


# ============================================================================
# Config Utils (embedded from models/config_utils.py)
# ============================================================================

def _dict_to_namespace(d):
    """Recursively convert a dict to SimpleNamespace for attribute access."""
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = _dict_to_namespace(v)
        return SimpleNamespace(**d)
    if isinstance(d, list):
        return [_dict_to_namespace(item) for item in d]
    return d


def load_config(model_name):
    """从 HuggingFace 缓存加载 config.json，不联网下载。"""
    config_path = hf_hub_download(
        repo_id=model_name, filename="config.json", local_files_only=True
    )
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return _dict_to_namespace(config_dict)


# ============================================================================
# Weight Utils (embedded from models/weight_utils.py)
# ============================================================================

def download_state_dict(model_name):
    """从 HuggingFace 缓存加载模型权重，不联网下载。"""
    config_path = hf_hub_download(
        repo_id=model_name, filename="config.json", local_files_only=True
    )
    snapshot_dir = os.path.dirname(config_path)
    repo_files = os.listdir(snapshot_dir)

    safetensor_files = [f for f in repo_files if f.endswith(".safetensors")]
    bin_files = [f for f in repo_files if f.endswith(".bin") and "pytorch_model" in f]

    if safetensor_files:
        from safetensors.torch import load_file
        state_dict = {}
        for sf in sorted(safetensor_files):
            path = os.path.join(snapshot_dir, sf)
            state_dict.update(load_file(path))
        return state_dict
    elif bin_files:
        state_dict = {}
        for bf in sorted(bin_files):
            path = os.path.join(snapshot_dir, bf)
            state_dict.update(torch.load(path, map_location="cpu", weights_only=True))
        return state_dict
    else:
        path = os.path.join(snapshot_dir, "pytorch_model.bin")
        return torch.load(path, map_location="cpu", weights_only=True)


# ============================================================================
# GPT-2 Model Implementation (embedded from models/gpt2.py)
# ============================================================================

def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class GPT2Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.layer_idx = layer_idx

        # GPT-2 uses Conv1D which is a linear with transposed weight (nf, nx)
        # We use nn.Linear and transpose weights during loading
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.size()

        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = gelu_new(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if getattr(config, "n_inner", None) is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        hidden_states = feed_forward_output + residual

        return hidden_states


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embd

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, attention_mask=None):
        bsz, seq_len = input_ids.size()
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPT2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return logits

    @classmethod
    def from_pretrained(cls, model_name):
        config = load_config(model_name)
        model = cls(config)

        hf_sd = download_state_dict(model_name)
        new_sd = {}
        for k, v in hf_sd.items():
            # Conv1D weights need transposing for nn.Linear
            # Conv1D stores weight as (in_features, out_features), nn.Linear as (out_features, in_features)
            new_key = k
            if new_key.startswith("transformer."):
                pass  # keep as is
            else:
                continue

            # FIXED: Only transpose Conv1D weights (.c_attn, .c_proj, .c_fc)
            # Embeddings (.wte, .wpe) should NOT be transposed
            if any(suffix in k for suffix in [".c_attn.weight", ".c_proj.weight", ".c_fc.weight"]):
                # These are Conv1D weights that need transposing
                new_sd[new_key] = v.t()
            else:
                new_sd[new_key] = v

        # Handle lm_head weight tying (it may or may not be in state_dict)
        if "lm_head.weight" in hf_sd:
            new_sd["lm_head.weight"] = hf_sd["lm_head.weight"]

        model.load_state_dict(new_sd, strict=False)
        # Ensure weight tying after loading
        model.lm_head.weight = model.transformer.wte.weight
        return model


# ============================================================================
# Benchmark Interface
# ============================================================================

class Model(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model = GPT2ForCausalLM.from_pretrained(model_name)

    def forward(self, x):
        return self.model(x)


model_name = "gpt2"
config = load_config(model_name)
vocab_size = config.vocab_size
sequence_length = 1023
batch_size = 1


def get_inputs():
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]


def get_init_inputs():
    return [model_name, config]
