"""Self-contained BigBird benchmark: batch_size=32, sequence_length=256

BigBird implementation with block-sparse attention for efficient long sequence modeling.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Static BigBirdConfig
# ============================================================================

class BigBirdConfig:
    """Hardcoded configuration for google/bigbird-roberta-base."""
    def __init__(self):
        self.vocab_size = 50358
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 4096
        self.type_vocab_size = 2
        self.initializer_range = 0.02
        self.layer_norm_eps = 1e-12
        self.pad_token_id = 0
        self.bos_token_id = 101
        self.eos_token_id = 102
        self.position_embedding_type = "absolute"
        # BigBird specific
        self.attention_type = "block_sparse"
        self.block_size = 64
        self.num_random_blocks = 3
        self.rescale_embeddings = False
        self.use_bias = True  # Aligned with official BigBirdConfig


# ============================================================================
# BigBird Model Components
# ============================================================================

class BigBirdEmbeddings(nn.Module):
    """BigBird embeddings with optional rescaling by sqrt(hidden_size)."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                            padding_idx=getattr(config, "pad_token_id", 0))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        self.rescale_embeddings = getattr(config, "rescale_embeddings", False)
        self.hidden_size = config.hidden_size

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        bsz, seq_len = input_ids.size()

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]
        if token_type_ids is None:
            token_type_ids = torch.zeros(bsz, seq_len, dtype=torch.long, device=input_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size ** 0.5)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Aligned with official HF BigBirdEmbeddings: add -> dropout -> LayerNorm
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BigBirdSelfAttention(nn.Module):
    """Standard full self-attention (used when sequence is short or attention_type='original_full')."""

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        use_bias = getattr(config, "use_bias", True)
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=use_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.size()

        query_layer = self.query(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key_layer = self.key(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = self.value(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        attn_weights = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_layer = torch.matmul(attn_weights, value_layer)
        context_layer = context_layer.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()

        return (context_layer,)


class BigBirdBlockSparseAttention(nn.Module):
    """Block-sparse attention with 5-part computation:
    1. First global block
    2. Second block
    3. Middle sliding+random blocks
    4. Second-last block
    5. Last global block

    Features:
    - Block size: 64
    - Random block selection with np.random.seed for determinism
    - Padding to block_size multiples
    """

    def __init__(self, config, seed=None):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_random_blocks = config.num_random_blocks
        self.block_size = config.block_size
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        use_bias = getattr(config, "use_bias", True)
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=use_bias)
        self.seed = seed

    def torch_bmm_nd(self, inp_1, inp_2, ndim=4):
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(
            inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1])
        )

    def torch_bmm_nd_transpose(self, inp_1, inp_2, ndim=4):
        return torch.bmm(
            inp_1.reshape((-1,) + inp_1.shape[-2:]),
            inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2),
        ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))

    def _bigbird_block_rand_mask(self, from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks):
        """Generate random block attention indices with deterministic seeding."""
        num_blocks = from_seq_length // from_block_size - 2
        if num_blocks <= 0:
            return np.zeros((0, num_rand_blocks), dtype=np.int32)

        rand_attn = np.zeros((num_blocks, num_rand_blocks), dtype=np.int32)
        if not self.training:
            return rand_attn

        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        r = num_rand_blocks

        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                elif (end + 1) == last:
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                else:
                    rand_attn[i - 1, :] = np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1:last]))
                    )[:r]
        return rand_attn

    def forward(self, hidden_states, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask,
                output_attentions=False):
        bsz, seq_len, _ = hidden_states.size()
        n_heads = self.num_attention_heads
        head_dim = self.attention_head_size
        block_size = self.block_size
        n_rand_blocks = self.num_random_blocks

        query_layer = self.query(hidden_states).view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
        key_layer = self.key(hidden_states).view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
        value_layer = self.value(hidden_states).view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)

        rsqrt_d = 1.0 / math.sqrt(head_dim)
        attn_mask_penalty = -10000.0

        # Block views
        num_blocks = seq_len // block_size
        blocked_query = query_layer.view(bsz, n_heads, num_blocks, block_size, head_dim)
        blocked_key = key_layer.view(bsz, n_heads, num_blocks, block_size, head_dim)
        blocked_value = value_layer.view(bsz, n_heads, num_blocks, block_size, head_dim)

        # Get random attention indices with deterministic seed
        if self.seed is not None:
            np.random.seed(self.seed)
        rand_attn = self._bigbird_block_rand_mask(seq_len, seq_len, block_size, block_size, n_rand_blocks)
        rand_attn = torch.tensor(rand_attn, dtype=torch.long, device=hidden_states.device)
        rand_attn = rand_attn.unsqueeze(0).unsqueeze(0).expand(bsz, n_heads, -1, -1)

        # Random mask for softmax
        rand_mask = self._create_rand_mask_from_inputs(
            from_blocked_mask, to_blocked_mask, rand_attn, n_heads, n_rand_blocks, bsz, seq_len, block_size
        )

        # Gather random key/value blocks
        gathered_key = self._gather_blocks(blocked_key, rand_attn, bsz, n_heads, head_dim, block_size)
        gathered_value = self._gather_blocks(blocked_value, rand_attn, bsz, n_heads, head_dim, block_size)

        # ===== 1st PART: first block (global) =====
        first_product = self.torch_bmm_nd_transpose(blocked_query[:, :, 0], key_layer, ndim=4)
        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = F.softmax(first_product, dim=-1)
        first_context = self.torch_bmm_nd(first_attn_weights, value_layer, ndim=4)
        first_context = first_context.unsqueeze(2)

        # ===== 2nd PART: second block =====
        second_key = torch.cat([
            blocked_key[:, :, 0], blocked_key[:, :, 1], blocked_key[:, :, 2],
            blocked_key[:, :, -1], gathered_key[:, :, 0]
        ], dim=2)
        second_value = torch.cat([
            blocked_value[:, :, 0], blocked_value[:, :, 1], blocked_value[:, :, 2],
            blocked_value[:, :, -1], gathered_value[:, :, 0]
        ], dim=2)
        second_product = self.torch_bmm_nd_transpose(blocked_query[:, :, 1], second_key, ndim=4)
        second_seq_pad = torch.cat([
            to_mask[:, :, :, :3 * block_size],
            to_mask[:, :, :, -block_size:],
            to_mask.new_ones([bsz, 1, 1, n_rand_blocks * block_size]),
        ], dim=3)
        second_rand_pad = torch.cat([
            rand_mask.new_ones([bsz, n_heads, block_size, 4 * block_size]),
            rand_mask[:, :, 0],
        ], dim=3)
        second_product = second_product * rsqrt_d
        second_product += (1.0 - torch.minimum(second_seq_pad, second_rand_pad)) * attn_mask_penalty
        second_attn_weights = F.softmax(second_product, dim=-1)
        second_context = self.torch_bmm_nd(second_attn_weights, second_value, ndim=4)
        second_context = second_context.unsqueeze(2)

        # ===== 3rd PART: middle blocks (sliding window + random + global) =====
        if num_blocks > 4:
            # Sliding window keys: [block i-1, block i, block i+1]
            exp_blocked_key = torch.cat([
                blocked_key[:, :, 1:-3], blocked_key[:, :, 2:-2], blocked_key[:, :, 3:-1]
            ], dim=3)
            exp_blocked_value = torch.cat([
                blocked_value[:, :, 1:-3], blocked_value[:, :, 2:-2], blocked_value[:, :, 3:-1]
            ], dim=3)

            middle_query = blocked_query[:, :, 2:-2]
            # Global (first + last) + sliding + random
            inner_band_product = self.torch_bmm_nd_transpose(middle_query, exp_blocked_key, ndim=5)
            inner_band_product = inner_band_product * rsqrt_d

            # First/last block attention
            first_band_product = torch.einsum("bhlqd,bhkd->bhlqk", middle_query, blocked_key[:, :, 0])
            first_band_product = first_band_product * rsqrt_d
            last_band_product = torch.einsum("bhlqd,bhkd->bhlqk", middle_query, blocked_key[:, :, -1])
            last_band_product = last_band_product * rsqrt_d

            # Random block attention
            inner_rand_keys = gathered_key[:, :, 1:-1]
            rand_band_product = torch.einsum("bhlqd,bhlkd->bhlqk", middle_query, inner_rand_keys)
            rand_band_product = rand_band_product * rsqrt_d

            # Masks
            inner_band_product += (1.0 - band_mask) * attn_mask_penalty
            first_band_product += (1.0 - to_mask[:, :, :, :block_size].unsqueeze(2).expand(-1, -1, middle_query.shape[2], -1, -1)) * attn_mask_penalty
            last_band_product += (1.0 - to_mask[:, :, :, -block_size:].unsqueeze(2).expand(-1, -1, middle_query.shape[2], -1, -1)) * attn_mask_penalty
            rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty

            # Combine and softmax
            band_product = torch.cat([
                first_band_product, inner_band_product, last_band_product, rand_band_product
            ], dim=-1)
            attn_weights = F.softmax(band_product, dim=-1)

            # Attend to values
            first_attn = attn_weights[:, :, :, :, :block_size]
            inner_attn = attn_weights[:, :, :, :, block_size:block_size + 3 * block_size]
            last_attn = attn_weights[:, :, :, :, block_size + 3 * block_size:2 * block_size + 3 * block_size]
            rand_attn_w = attn_weights[:, :, :, :, 2 * block_size + 3 * block_size:]

            first_context_mid = torch.einsum("bhlqk,bhkd->bhlqd", first_attn, blocked_value[:, :, 0])
            inner_context = self.torch_bmm_nd(inner_attn, exp_blocked_value, ndim=5)
            last_context_mid = torch.einsum("bhlqk,bhkd->bhlqd", last_attn, blocked_value[:, :, -1])
            rand_context = torch.einsum("bhlqk,bhlkd->bhlqd", rand_attn_w, gathered_value[:, :, 1:-1])

            middle_context = first_context_mid + inner_context + last_context_mid + rand_context
        else:
            middle_context = torch.zeros(
                bsz, n_heads, 0, block_size, head_dim, device=hidden_states.device, dtype=hidden_states.dtype
            )

        # ===== 4th PART: second-to-last block =====
        second_last_key = torch.cat([
            blocked_key[:, :, 0], blocked_key[:, :, -3], blocked_key[:, :, -2],
            blocked_key[:, :, -1], gathered_key[:, :, -1]
        ], dim=2)
        second_last_value = torch.cat([
            blocked_value[:, :, 0], blocked_value[:, :, -3], blocked_value[:, :, -2],
            blocked_value[:, :, -1], gathered_value[:, :, -1]
        ], dim=2)
        second_last_product = self.torch_bmm_nd_transpose(blocked_query[:, :, -2], second_last_key, ndim=4)
        second_last_seq_pad = torch.cat([
            to_mask[:, :, :, :block_size],
            to_mask[:, :, :, -3 * block_size:],
            to_mask.new_ones([bsz, 1, 1, n_rand_blocks * block_size]),
        ], dim=3)
        second_last_rand_pad = torch.cat([
            rand_mask.new_ones([bsz, n_heads, block_size, 4 * block_size]),
            rand_mask[:, :, -1],
        ], dim=3)
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - torch.minimum(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
        second_last_attn_weights = F.softmax(second_last_product, dim=-1)
        second_last_context = self.torch_bmm_nd(second_last_attn_weights, second_last_value, ndim=4)
        second_last_context = second_last_context.unsqueeze(2)

        # ===== 5th PART: last block (global) =====
        last_product = self.torch_bmm_nd_transpose(blocked_query[:, :, -1], key_layer, ndim=4)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * attn_mask_penalty
        last_attn_weights = F.softmax(last_product, dim=-1)
        last_context = self.torch_bmm_nd(last_attn_weights, value_layer, ndim=4)
        last_context = last_context.unsqueeze(2)

        # Combine all parts
        context_layer = torch.cat([
            first_context, second_context, middle_context, second_last_context, last_context
        ], dim=2)
        context_layer = context_layer.view(bsz, n_heads, seq_len, head_dim)
        context_layer = context_layer * from_mask.view(bsz, 1, seq_len, 1)
        context_layer = context_layer.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()

        return (context_layer,)

    def _gather_blocks(self, blocked, rand_attn, bsz, n_heads, head_dim, block_size):
        """Gather random blocks based on rand_attn indices."""
        num_rand = rand_attn.shape[-1]
        num_inner = rand_attn.shape[2]
        # Expand rand_attn for gathering
        rand_attn_expanded = rand_attn.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, block_size, head_dim)
        blocked_expanded = blocked.unsqueeze(2).expand(-1, -1, num_inner, -1, -1, -1)
        gathered = torch.gather(blocked_expanded, 3, rand_attn_expanded)
        # Reshape to [bsz, n_heads, num_inner, n_rand_blocks * block_size, head_dim]
        gathered = gathered.reshape(bsz, n_heads, num_inner, num_rand * block_size, head_dim)
        return gathered

    def _create_rand_mask_from_inputs(self, from_blocked_mask, to_blocked_mask, rand_attn, n_heads, n_rand_blocks,
                                       bsz, seq_len, block_size):
        """Create random attention mask."""
        num_inner = rand_attn.shape[2]
        # For each inner block, gather the to_blocked_mask for the random blocks
        rand_attn_expanded = rand_attn.unsqueeze(-1).expand(-1, -1, -1, -1, block_size)
        to_blocked_expanded = to_blocked_mask.unsqueeze(1).unsqueeze(2).expand(-1, n_heads, num_inner, -1, -1)
        rand_mask = torch.gather(to_blocked_expanded, 3, rand_attn_expanded)
        # Shape: [bsz, n_heads, num_inner, n_rand_blocks, block_size]
        rand_mask = rand_mask.reshape(bsz, n_heads, num_inner, n_rand_blocks * block_size)
        # Expand for query block_size
        rand_mask = rand_mask.unsqueeze(-2).expand(-1, -1, -1, block_size, -1)
        return rand_mask


class BigBirdSelfOutput(nn.Module):
    """Post-LN structure: dense → dropout → add residual → LayerNorm."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BigBirdAttention(nn.Module):
    """Aligned with official HF: uses original_full when band_mask is None (short seq)."""
    def __init__(self, config, seed=None):
        super().__init__()
        self.attention_type = config.attention_type
        self.config = config
        self.seed = seed
        self.self_full = BigBirdSelfAttention(config)
        if self.attention_type == "block_sparse":
            self.self_block = BigBirdBlockSparseAttention(config, seed)
        else:
            self.self_block = None
        self.output = BigBirdSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, band_mask=None, from_mask=None,
                to_mask=None, from_blocked_mask=None, to_blocked_mask=None):
        use_full = (self.attention_type == "original_full" or
                    (band_mask is None and from_mask is None))
        if use_full:
            self_outputs = self.self_full(hidden_states, attention_mask=attention_mask)
        else:
            # Cast masks to hidden_states dtype
            if band_mask is not None:
                band_mask = band_mask.to(hidden_states.dtype)
            if from_mask is not None:
                from_mask = from_mask.to(hidden_states.dtype)
            if to_mask is not None:
                to_mask = to_mask.to(hidden_states.dtype)
            self_outputs = self.self_block(
                hidden_states, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask
            )

        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output


class BigBirdIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class BigBirdOutput(nn.Module):
    """Post-LN structure: dense → dropout → add residual → LayerNorm."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BigBirdLayer(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()
        self.attention = BigBirdAttention(config, seed=seed)
        self.intermediate = BigBirdIntermediate(config)
        self.output = BigBirdOutput(config)

    def forward(self, hidden_states, attention_mask=None, band_mask=None, from_mask=None,
                to_mask=None, blocked_encoder_mask=None):
        attention_output = self.attention(
            hidden_states, attention_mask=attention_mask,
            band_mask=band_mask, from_mask=from_mask, to_mask=to_mask,
            from_blocked_mask=blocked_encoder_mask, to_blocked_mask=blocked_encoder_mask,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BigBirdEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type
        self.layer = nn.ModuleList([
            BigBirdLayer(config, seed=layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states, attention_mask=None, band_mask=None, from_mask=None,
                to_mask=None, blocked_encoder_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states, attention_mask=attention_mask,
                band_mask=band_mask, from_mask=from_mask, to_mask=to_mask,
                blocked_encoder_mask=blocked_encoder_mask,
            )
        return hidden_states


class BigBirdModel(nn.Module):
    """BigBird encoder with padding to block_size multiples and mask creation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type
        self.block_size = config.block_size
        self.embeddings = BigBirdEmbeddings(config)
        self.encoder = BigBirdEncoder(config)

    def _pad_to_block_size(self, input_ids, attention_mask, token_type_ids):
        """Pad sequence to block_size multiples (block_size=64)."""
        block_size = self.block_size
        bsz, seq_len = input_ids.shape
        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            pad_token_id = getattr(self.config, "pad_token_id", 0)
            input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
            attention_mask = F.pad(attention_mask, (0, padding_len), value=0)
            if token_type_ids is not None:
                token_type_ids = F.pad(token_type_ids, (0, padding_len), value=0)
        return padding_len, input_ids, attention_mask, token_type_ids

    @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask, block_size):
        """Create masks for block-sparse attention:
        - blocked_encoder_mask: [bsz, num_blocks, block_size]
        - band_mask: sliding window mask for middle blocks
        - from_mask: [bsz, 1, seq_len, 1]
        - to_mask: [bsz, 1, 1, seq_len]
        """
        bsz, seq_len = attention_mask.size()
        blocked_encoder_mask = attention_mask.view(bsz, seq_len // block_size, block_size)

        # Band mask for sliding window attention
        from_blocked = blocked_encoder_mask
        to_blocked = blocked_encoder_mask
        exp_blocked_to_pad = torch.cat([
            to_blocked[:, 1:-3], to_blocked[:, 2:-2], to_blocked[:, 3:-1]
        ], dim=2)
        band_mask = torch.einsum("blq,blk->blqk", from_blocked[:, 2:-2].float(), exp_blocked_to_pad.float())
        band_mask = band_mask.unsqueeze(1)

        from_mask = attention_mask.view(bsz, 1, seq_len, 1).float()
        to_mask = attention_mask.view(bsz, 1, 1, seq_len).float()

        return blocked_encoder_mask, band_mask, from_mask, to_mask

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        bsz, orig_seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones(bsz, orig_seq_len, dtype=torch.long, device=input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(bsz, orig_seq_len, dtype=torch.long, device=input_ids.device)

        min_seq_for_block_sparse = 5 * self.block_size
        use_block_sparse = (self.attention_type == "block_sparse" and
                            orig_seq_len >= min_seq_for_block_sparse)
        if use_block_sparse:
            padding_len, input_ids, attention_mask, token_type_ids = self._pad_to_block_size(
                input_ids, attention_mask, token_type_ids
            )
            blocked_encoder_mask, band_mask, from_mask, to_mask = self.create_masks_for_block_sparse_attn(
                attention_mask, self.block_size
            )
            extended_attention_mask = None
        else:
            padding_len = 0
            blocked_encoder_mask = band_mask = from_mask = to_mask = None
            extended_attention_mask = attention_mask[:, None, None, :].float()
            # Use torch.finfo().min instead of float('-inf') for precision alignment
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(extended_attention_mask.dtype).min

        hidden_states = self.embeddings(input_ids, token_type_ids=token_type_ids)

        encoder_output = self.encoder(
            hidden_states, attention_mask=extended_attention_mask,
            band_mask=band_mask, from_mask=from_mask, to_mask=to_mask,
            blocked_encoder_mask=blocked_encoder_mask,
        )

        # Remove padding
        if padding_len > 0:
            encoder_output = encoder_output[:, :orig_seq_len]

        return encoder_output


class BigBirdPredictionHeadTransform(nn.Module):
    """Prediction head transform: dense → gelu → LayerNorm."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BigBirdLMPredictionHead(nn.Module):
    """MLM prediction head: transform → decoder."""

    def __init__(self, config):
        super().__init__()
        self.transform = BigBirdPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BigBirdForMaskedLM(nn.Module):
    """BigBird for Masked Language Modeling with weight tying."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BigBirdModel(config)
        self.cls = BigBirdLMPredictionHead(config)
        # Weight tying: share embedding weights with output layer
        self.cls.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prediction_scores = self.cls(sequence_output)
        return prediction_scores


# ============================================================================
# Benchmark setup
# ============================================================================

class Model(torch.nn.Module):
    """Benchmark model wrapper for BigBird."""
    def __init__(self, config):
        super().__init__()
        # Initialize model with static config (no weight loading)
        self.model = BigBirdForMaskedLM(config)

    def forward(self, x):
        # Forward pass: input_ids -> prediction scores
        return self.model(x)


# Static configuration instance
config = BigBirdConfig()
vocab_size = config.vocab_size
sequence_length = 256
batch_size = 32


def get_inputs():
    """Generate benchmark inputs: random input_ids of shape (batch_size, sequence_length)."""
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]


def get_init_inputs():
    """Return initialization inputs for Model.__init__()."""
    return [config]
