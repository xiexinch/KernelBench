"""Self-contained BART benchmark: facebook/bart-large with bs=1024, seq=32"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


# ============================================================================
# Configuration Utilities
# ============================================================================

def _dict_to_namespace(d):
    """Recursively convert a dict to SimpleNamespace for attribute access."""
    from types import SimpleNamespace
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
# BART Decoder-Only Implementation
# ============================================================================

class BartLearnedPositionalEmbedding(nn.Embedding):
    """Learned positional embedding with offset=2 (BART convention)."""

    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0, position_ids=None):
        if position_ids is None:
            bsz, seq_len = input_ids.shape[:2]
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_len,
                dtype=torch.long, device=self.weight.device
            ).expand(bsz, -1)
        return super().forward(position_ids + self.offset)


class BartScaledWordEmbedding(nn.Embedding):
    """Word embedding with optional scaling by sqrt(d_model)."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale=1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale


class BartAttention(nn.Module):
    """BART attention with Q,K,V,out projections all nn.Linear with bias=True."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = dropout

    def forward(self, hidden_states, attention_mask=None):
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class BartDecoderLayer(nn.Module):
    """BART decoder layer with Post-LN structure and GELU activation."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout

        # Activation function (GELU for BART)
        act_fn = getattr(config, "activation_function", "gelu")
        if act_fn == "gelu":
            self.activation_fn = F.gelu
        elif act_fn == "relu":
            self.activation_fn = F.relu
        else:
            self.activation_fn = F.gelu

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        # Post-LN: attn -> dropout -> residual -> LayerNorm
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # FFN: fc1 -> act -> dropout -> fc2 -> dropout -> residual -> LayerNorm
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class BartDecoder(nn.Module):
    """BART decoder with layernorm_embedding after embed+pos combination."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.padding_idx = config.pad_token_id

        # Optional embedding scaling by sqrt(d_model) if config.scale_embedding=True
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale
        )
        # BartLearnedPositionalEmbedding with offset=2
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model
        )
        # layernorm_embedding after embed+pos combination
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layers = nn.ModuleList([
            BartDecoderLayer(config, layer_idx=i)
            for i in range(config.decoder_layers)
        ])

    def forward(self, input_ids, attention_mask=None):
        inputs_embeds = self.embed_tokens(input_ids)
        positions = self.embed_positions(input_ids)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # Create causal mask
        bsz, seq_len = input_ids.shape
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.to(hidden_states.dtype) * torch.finfo(hidden_states.dtype).min

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=causal_mask)

        return hidden_states


class BartForCausalLM(nn.Module):
    """BART for causal language modeling with weight tying."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        config.is_decoder = True
        config.is_encoder_decoder = False
        self.decoder = BartDecoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
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
        self.model = BartForCausalLM.from_pretrained(model_name)

    def forward(self, x):
        return self.model(x)


model_name = "facebook/bart-large"
config = load_config(model_name)
vocab_size = config.vocab_size
sequence_length = 32
batch_size = 1024


def get_inputs():
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]


def get_init_inputs():
    return [model_name, config]
