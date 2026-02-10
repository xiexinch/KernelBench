"""Self-contained OPT-1.3B benchmark: batch_size=32, sequence_length=256

This file contains a complete PyTorch implementation of OPT with key features:
- Learned positional embeddings with offset=2 trick
- Position IDs from attention mask: cumsum(attn_mask) * attn_mask - 1
- Pre-LN structure (do_layer_norm_before=True for opt-1.3b)
- Query scaling by head_dim^-0.5
- Simplified softmax (no explicit dtype conversion)
- Causal mask uses torch.finfo().min
- Optional embed projection (project_in/project_out) when word_embed_proj_dim != hidden_size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Static Configuration
# ============================================================================

class OPTConfig:
    """Static hardcoded OPT-1.3B configuration."""
    def __init__(self):
        self.vocab_size = 50272
        self.hidden_size = 768
        self.num_hidden_layers = 24
        self.ffn_dim = 3072
        self.max_position_embeddings = 2048
        self.num_attention_heads = 12
        self.word_embed_proj_dim = 768
        self.dropout = 0.1
        self.activation_function = "relu"
        self.do_layer_norm_before = True
        self.enable_bias = True
        self.layer_norm_elementwise_affine = True
        self.pad_token_id = 1


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
            # Compute position IDs from attention mask: cumsum(mask) - 1, then add offset
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

        enable_bias = config.enable_bias
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

        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: Q @ K^T
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax without explicit dtype conversion
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Attention output: softmax @ V
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

        enable_bias = config.enable_bias
        layer_norm_elementwise_affine = config.layer_norm_elementwise_affine

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=enable_bias)
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=layer_norm_elementwise_affine
        )

        # Activation function (ReLU for opt-1.3b)
        if config.activation_function == "relu":
            self.activation_fn = F.relu
        elif config.activation_function == "gelu":
            self.activation_fn = F.gelu
        else:
            self.activation_fn = F.relu

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states

        # Pre-LN: layer norm before attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self-attention
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Post-LN (if not pre-LN)
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Feed-forward network
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

        # Post-LN (if not pre-LN)
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class OPTDecoder(nn.Module):
    """OPT decoder with optional embed projection."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.padding_idx = config.pad_token_id

        # Token and position embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        # Optional embed projection when word_embed_proj_dim != hidden_size
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_in = None
            self.project_out = None

        # Final layer norm (only for pre-LN)
        if config.do_layer_norm_before:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        # Decoder layers
        self.layers = nn.ModuleList([
            OPTDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids, attention_mask=None):
        # Embed input tokens
        inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            bsz, seq_len = input_ids.shape
            attention_mask = torch.ones(bsz, seq_len, device=input_ids.device)

        # Position embeddings from attention mask
        pos_embeds = self.embed_positions(attention_mask)

        # Optional embed projection
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        # Create causal mask: 1 for future positions, 0 for past/current
        bsz, seq_len = input_ids.shape
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1
        )
        # Unsqueeze for batch and num_heads dimensions
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        # Use torch.finfo().min instead of float('-inf') for numerical stability
        causal_mask = causal_mask.to(hidden_states.dtype) * torch.finfo(hidden_states.dtype).min

        # Apply decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=causal_mask)

        # Final layer norm (if pre-LN)
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        # Optional embed projection output
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        return hidden_states


class OPTForCausalLM(nn.Module):
    """OPT for causal language modeling with weight tying."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = OPTDecoder(config)
        # LM head for next token prediction
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        # Weight tying: share weights between embedding and output layer
        self.lm_head.weight = self.decoder.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None):
        # Get decoder output
        hidden_states = self.decoder(input_ids, attention_mask=attention_mask)
        # Compute logits
        logits = self.lm_head(hidden_states)
        return logits


# ============================================================================
# Benchmark Setup
# ============================================================================

class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize OPT model with hardcoded config
        self.model = OPTForCausalLM(config)

    def forward(self, x):
        return self.model(x)


# Hardcoded config and batch parameters
config = OPTConfig()
vocab_size = config.vocab_size
sequence_length = 256
batch_size = 32


def get_inputs():
    # Generate random input token IDs
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]


def get_init_inputs():
    # Return config for Model initialization
    return [config]
