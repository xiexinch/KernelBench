"""
GPT-2 pure PyTorch expanded implementation (no transformers dependency).
Batch size=1, sequence_length=1023.
Interface consistent with level4/16_gpt2_bs1_seq1023.py, structure aligned with HuggingFace GPT2LMHeadModel.

Fixed precision alignment issues:
- Use hardcoded GPT2ConfigExpanded class instead of runtime config loading
- Use torch.finfo().min for causal mask instead of float("-inf")
- Simplify softmax without explicit dtype conversion
- Update Model interface to accept only config parameter
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2ConfigExpanded:
    """GPT-2 configuration (consistent with openai-community/gpt2 defaults)"""
    vocab_size = 50257
    n_positions = 1024
    n_embd = 768
    n_layer = 12
    n_head = 12
    n_inner = 3072
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    scale_attn_weights = True
    scale_attn_by_inverse_layer_idx = False
    hidden_size = 768
    max_position_embeddings = 1024
    num_attention_heads = 12
    num_hidden_layers = 12


def _gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class _GPT2Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        # Fixed: Register causal mask as buffer for consistent dtype handling
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.size()
        q, k, v = self.c_attn(hidden_states).split(self.split_size, dim=2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(v.size(-1)) ** 0.5)
        if self.scale_attn_by_inverse_layer_idx and self.layer_idx is not None:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        causal_mask = self.bias[:, :, seq_len - q.size(2) : seq_len, :seq_len]
        # Fixed dtype mismatch: use torch.finfo().min for causal mask instead of float("-inf")
        mask_value = torch.full([], torch.finfo(attn_weights.dtype).min, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        # Fixed softmax: simplified without explicit dtype conversion
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        attn_output = self.c_proj(attn_output)
        return self.resid_dropout(attn_output)


class _GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = _gelu_new(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)


class _GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = _GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = _GPT2MLP(inner_dim, config)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class _GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([_GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, attention_mask=None):
        device = input_ids.device
        bsz, seq_len = input_ids.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        return self.ln_f(hidden_states)


class _GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = _GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask)
        return self.lm_head(hidden_states)


# Configuration and interface (aligned with level4)
config = GPT2ConfigExpanded()
vocab_size = config.vocab_size
sequence_length = 1023
batch_size = 1


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = _GPT2LMHeadModel(config)

    def forward(self, x):
        return self.model(x)


def get_inputs():
    return [torch.randint(0, vocab_size, (batch_size, sequence_length))]


def get_init_inputs():
    return [config]
