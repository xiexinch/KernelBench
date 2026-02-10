"""
Self-contained GPT-Neo benchmark: batch_size=32, sequence_length=256
This file embeds the complete GPT-Neo model implementation with no external model dependencies.
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
# EMBEDDED: config_utils functionality
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
# EMBEDDED: weight_utils functionality
# ============================================================================

def download_state_dict(model_name):
    """从 HuggingFace 缓存加载模型权重，不联网下载。"""
    # 通过 config.json 的缓存路径得到 snapshot 目录，避免使用 list_repo_files（旧版无 local_files_only）
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
# EMBEDDED: GPT-Neo model implementation
# ============================================================================

def gelu_new(x):
    """GPT-Neo uses the approximate GELU activation function."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class GPTNeoSelfAttention(nn.Module):
    """
    GPT-Neo self-attention with support for both global and local (windowed) attention.
    Uses nn.Linear projections (not Conv1D like GPT-2).
    """
    def __init__(self, config, attention_type, layer_idx=None):
        super().__init__()
        self.config = config
        self.attention_type = attention_type

        # Create causal mask
        max_positions = config.max_position_embeddings
        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
            1, 1, max_positions, max_positions
        )

        # For local attention, use XOR-based windowed causal mask
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, -config.window_size))

        self.register_buffer("bias", bias, persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads

        # GPT-Neo uses nn.Linear (not Conv1D)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention in float32 for stability
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # Apply causal/local mask
        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(causal_mask, attn_weights, torch.tensor(mask_value, device=attn_weights.device))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class GPTNeoAttention(nn.Module):
    """Wrapper for GPT-Neo attention that selects attention type based on layer index."""
    def __init__(self, config, layer_idx=0):
        super().__init__()
        # Mixed global/local attention: alternating pattern based on config
        self.attention_type = config.attention_layers[layer_idx]
        self.attention = GPTNeoSelfAttention(config, self.attention_type, layer_idx)

    def forward(self, hidden_states):
        return self.attention(hidden_states)


class GPTNeoMLP(nn.Module):
    """GPT-Neo MLP with GELU activation."""
    def __init__(self, intermediate_size, config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, config.hidden_size)
        self.act = gelu_new
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTNeoBlock(nn.Module):
    """GPT-Neo transformer block with pre-norm architecture."""
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = getattr(config, "intermediate_size", None) or 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTNeoAttention(config, layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPTNeoMLP(inner_dim, config)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        hidden_states = feed_forward_output + residual

        return hidden_states


class GPTNeoModel(nn.Module):
    """GPT-Neo transformer model (without language modeling head)."""
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(float(config.embed_dropout))
        self.h = nn.ModuleList([GPTNeoBlock(config, layer_idx=i) for i in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.size()
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPTNeoForCausalLM(nn.Module):
    """
    GPT-Neo model for causal language modeling.
    Uses weight tying between lm_head and wte (token embeddings).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPTNeoModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Weight tying: lm_head shares weights with token embeddings
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids):
        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)
        return logits

    @classmethod
    def from_pretrained(cls, model_name):
        """Load pretrained GPT-Neo model from HuggingFace hub."""
        config = load_config(model_name)
        model = cls(config)

        hf_sd = download_state_dict(model_name)
        new_sd = {}
        for k, v in hf_sd.items():
            if k.startswith("transformer."):
                new_sd[k] = v

        if "lm_head.weight" in hf_sd:
            new_sd["lm_head.weight"] = hf_sd["lm_head.weight"]

        model.load_state_dict(new_sd, strict=False)
        # Ensure weight tying after loading
        model.lm_head.weight = model.transformer.wte.weight
        return model


# ============================================================================
# Benchmark interface (maintains same Model class interface)
# ============================================================================

class Model(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model = GPTNeoForCausalLM.from_pretrained(model_name)

    def forward(self, x):
        return self.model(x)


model_name = "EleutherAI/gpt-neo-2.7B"
config = load_config(model_name)
vocab_size = config.vocab_size
sequence_length = 256
batch_size = 32


def get_inputs():
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]


def get_init_inputs():
    return [model_name, config]
