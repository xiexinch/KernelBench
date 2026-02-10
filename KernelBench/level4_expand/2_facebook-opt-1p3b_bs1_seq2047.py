"""Self-contained OPT-1.3B benchmark: batch_size=1, sequence_length=2047

This file contains a complete PyTorch implementation of OPT with key features:
- Learned positional embeddings with offset=2 trick
- Position IDs from attention mask: cumsum(attn_mask) * attn_mask - 1
- Pre-LN structure (do_layer_norm_before=True for opt-1.3b)
- Query scaling by head_dim^-0.5
- Softmax in float32, cast back to query dtype
- Optional embed projection (project_in/project_out) when word_embed_proj_dim != hidden_size
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
# Configuration Loading
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
# Weight Loading
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
# OPT Model Components
# ============================================================================

class OPTLearnedPositionalEmbedding(nn.Embedding):
    """Learned positional embedding with offset=2 (BART/OPT convention)."""

    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            # Compute from attention mask: cumsum(attn_mask) * attn_mask - 1
            position_ids = torch.cumsum(attention_mask, dim=1).long() - 1
            position_ids = position_ids[:, past_key_values_length:]
        return super().forward(position_ids + self.offset)


class OPTAttention(nn.Module):
    """OPT attention with query scaling and float32 softmax."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # Query scaling by head_dim^-0.5
        self.scaling = self.head_dim ** -0.5

        enable_bias = getattr(config, "enable_bias", True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=enable_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=enable_bias)

    def forward(self, hidden_states, attention_mask=None):
        bsz, tgt_len, _ = hidden_states.size()

        # Query scaling
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax in float32, cast back to query dtype
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class OPTDecoderLayer(nn.Module):
    """OPT decoder layer with Pre-LN structure."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(config, layer_idx=layer_idx)
        # Pre-LN structure (do_layer_norm_before=True for opt-1.3b)
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout

        enable_bias = getattr(config, "enable_bias", True)
        layer_norm_elementwise_affine = getattr(config, "layer_norm_elementwise_affine", True)

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=enable_bias)
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=layer_norm_elementwise_affine
        )

        # Activation function
        act_fn_name = getattr(config, "activation_function", "relu")
        if act_fn_name == "relu":
            self.activation_fn = F.relu
        elif act_fn_name == "gelu":
            self.activation_fn = F.gelu
        else:
            self.activation_fn = F.relu

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states

        # Pre-LN: layer norm before attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # FFN
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # Pre-LN: layer norm before FFN
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class OPTDecoder(nn.Module):
    """OPT decoder with optional embed projection."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.padding_idx = getattr(config, "pad_token_id", 1)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        # Optional embed projection when word_embed_proj_dim != hidden_size
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_in = None
            self.project_out = None

        if config.do_layer_norm_before:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=getattr(config, "layer_norm_elementwise_affine", True)
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([
            OPTDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids, attention_mask=None):
        inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            bsz, seq_len = input_ids.shape
            attention_mask = torch.ones(bsz, seq_len, device=input_ids.device)

        # Position embeddings from attention mask
        pos_embeds = self.embed_positions(attention_mask)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        # Create causal mask
        bsz, seq_len = input_ids.shape
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.to(hidden_states.dtype) * torch.finfo(hidden_states.dtype).min

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=causal_mask)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        return hidden_states


class OPTForCausalLM(nn.Module):
    """OPT for causal language modeling with weight tying."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = OPTDecoder(config)
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.decoder.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.decoder(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return logits

    @classmethod
    def from_pretrained(cls, model_name):
        config = load_config(model_name)
        model = cls(config)

        hf_sd = download_state_dict(model_name)
        new_sd = {}
        for k, v in hf_sd.items():
            new_key = k
            # Map from HF naming: model.decoder.* -> decoder.*
            if new_key.startswith("model.decoder."):
                new_key = new_key[len("model."):]
            elif new_key == "lm_head.weight":
                new_sd[new_key] = v
                continue
            else:
                continue
            new_sd[new_key] = v

        if "lm_head.weight" in hf_sd:
            new_sd["lm_head.weight"] = hf_sd["lm_head.weight"]

        model.load_state_dict(new_sd, strict=False)
        model.lm_head.weight = model.decoder.embed_tokens.weight
        return model


# ============================================================================
# Benchmark Setup
# ============================================================================

class Model(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model = OPTForCausalLM.from_pretrained(model_name)

    def forward(self, x):
        return self.model(x)


model_name = "facebook/opt-1.3b"
config = load_config(model_name)
vocab_size = config.vocab_size
sequence_length = 2047
batch_size = 1


def get_inputs():
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]


def get_init_inputs():
    return [model_name, config]
