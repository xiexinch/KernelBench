"""Self-contained Reformer benchmark: google/reformer-enwik8, batch_size=1024, sequence_length=32"""

import math
import sys
from collections import namedtuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


# Named tuples for outputs
LSHSelfAttentionOutput = namedtuple("LSHSelfAttentionOutput", ["hidden_states", "attention_probs", "buckets"])
LocalSelfAttentionOutput = namedtuple("LocalSelfAttentionOutput", ["hidden_states", "attention_probs"])
ReformerOutput = namedtuple("ReformerOutput", ["attn_output", "hidden_states", "attention_probs", "buckets"])


def _stable_argsort(vector, dim):
    scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
    return torch.argsort(scaled_vector, dim=dim)


class AxialPositionEmbeddings(nn.Module):
    """Factorized 2D position embeddings: [64, 64] with dims [64, 192]"""
    def __init__(self, config):
        super().__init__()
        self.axial_pos_shape = config.axial_pos_shape  # e.g. [64, 64]
        self.axial_pos_embds_dim = config.axial_pos_embds_dim  # e.g. [64, 192]
        self.dropout = config.hidden_dropout_prob

        # Create parameter lists for each axis
        self.weights = nn.ParameterList()
        for axis_idx, (shape, dim) in enumerate(zip(self.axial_pos_shape, self.axial_pos_embds_dim)):
            self.weights.append(nn.Parameter(torch.zeros(shape, dim)))

    def forward(self, position_ids):
        batch_size = position_ids.shape[0]
        seq_len = position_ids.shape[1]

        # Compute full position embeddings from factored axial embeddings
        full_position_embeddings = self._compute_axial_embeddings(seq_len, position_ids.device)

        # Take only the positions we need
        position_embeddings = full_position_embeddings[:seq_len]
        position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply dropout during training
        if self.training and self.dropout > 0:
            position_embeddings = F.dropout(position_embeddings, p=self.dropout, training=True)

        return position_embeddings

    def _compute_axial_embeddings(self, seq_len, device):
        # Outer product of axial embeddings
        # weights[0]: [shape0, dim0], weights[1]: [shape1, dim1]
        # Result: [shape0 * shape1, dim0 + dim1]
        w0 = self.weights[0].to(device)  # [shape0, dim0]
        w1 = self.weights[1].to(device)  # [shape1, dim1]

        shape0, dim0 = w0.shape
        shape1, dim1 = w1.shape

        # Expand and concatenate: outer product along position dimensions
        w0_expanded = w0.unsqueeze(1).expand(-1, shape1, -1).reshape(shape0 * shape1, dim0)
        w1_expanded = w1.unsqueeze(0).expand(shape0, -1, -1).reshape(shape0 * shape1, dim1)

        return torch.cat([w0_expanded, w1_expanded], dim=-1)


class ReformerEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = AxialPositionEmbeddings(config)
        self.dropout = config.hidden_dropout_prob

    def forward(self, input_ids):
        bsz, seq_len = input_ids.size()
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class LSHSelfAttention(nn.Module):
    """LSH Self-Attention with random rotations (torch.manual_seed(hash_seed)) and bucket sorting"""
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        self.chunk_length = config.lsh_attn_chunk_length
        self.num_hashes = config.num_hashes
        self.num_buckets = config.num_buckets
        self.num_chunks_before = getattr(config, "lsh_num_chunks_before", 1)
        self.num_chunks_after = getattr(config, "lsh_num_chunks_after", 0)

        self.hash_seed = layer_idx  # Used for torch.manual_seed()

        # Shared Q/K projection
        self.query_key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        self.self_mask_value_float16 = torch.tensor(-1e3)
        self.self_mask_value_float32 = torch.tensor(-1e5)
        self.mask_value_float16 = torch.tensor(-1e4)
        self.mask_value_float32 = torch.tensor(-1e9)

        self.dropout = config.lsh_attention_probs_dropout_prob

    def forward(self, hidden_states, attention_mask=None, num_hashes=None, buckets=None,
                output_attentions=False):
        bsz, seq_len, _ = hidden_states.shape
        num_hashes = num_hashes if num_hashes is not None else self.num_hashes

        query_key_vectors = self.query_key(hidden_states)
        value_vectors = self.value(hidden_states)

        query_key_vectors = query_key_vectors.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)
        query_key_vectors = query_key_vectors.transpose(1, 2)
        value_vectors = value_vectors.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)
        value_vectors = value_vectors.transpose(1, 2)

        # For short sequences, use standard attention
        if seq_len <= self.chunk_length:
            return self._standard_attention(query_key_vectors, value_vectors, attention_mask, bsz, seq_len)

        # Length-normalize query_key
        query_key_vectors = self._len_and_dim_norm(query_key_vectors)

        # Hash vectors with seeded random rotations
        if buckets is None:
            buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)

        # Sort by buckets
        _, sorted_indices = torch.sort(buckets, dim=-1, stable=True)
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)

        # Expand for num_hashes
        query_key_vectors = query_key_vectors.unsqueeze(2).expand(-1, -1, num_hashes, -1, -1)
        query_key_vectors = query_key_vectors.reshape(bsz, self.num_attention_heads, -1, self.attention_head_size)
        value_vectors_expanded = value_vectors.unsqueeze(2).expand(-1, -1, num_hashes, -1, -1)
        value_vectors_expanded = value_vectors_expanded.reshape(bsz, self.num_attention_heads, -1, self.attention_head_size)

        sorted_qk = torch.gather(query_key_vectors, 2, sorted_indices_expanded)
        sorted_v = torch.gather(value_vectors_expanded, 2, sorted_indices_expanded)

        # Chunk sorted sequences
        total_len = sorted_qk.shape[2]
        chunk_len = self.chunk_length
        num_chunks = total_len // chunk_len

        sorted_qk = sorted_qk.reshape(bsz, self.num_attention_heads, num_chunks, chunk_len, self.attention_head_size)
        sorted_v = sorted_v.reshape(bsz, self.num_attention_heads, num_chunks, chunk_len, self.attention_head_size)
        sorted_buckets = torch.gather(buckets, 2, sorted_indices)
        sorted_buckets = sorted_buckets.reshape(bsz, self.num_attention_heads, num_chunks, chunk_len)

        # Look adjacent chunks
        sorted_qk_adj = self._look_adjacent(sorted_qk, self.num_chunks_before, self.num_chunks_after)
        sorted_v_adj = self._look_adjacent(sorted_v, self.num_chunks_before, self.num_chunks_after)
        sorted_buckets_adj = self._look_adjacent(sorted_buckets, self.num_chunks_before, self.num_chunks_after)

        # Compute attention
        query_vectors = sorted_qk
        key_vectors = sorted_qk_adj

        attn_weights = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # FIX: dtype-aware self-mask handling for precision alignment
        query_bucket_idx = sorted_buckets.unsqueeze(-1)
        key_bucket_idx = sorted_buckets_adj.unsqueeze(-2)
        self_mask = query_bucket_idx != key_bucket_idx
        if hidden_states.dtype == torch.float16:
            self_mask_value = self.self_mask_value_float16.to(hidden_states.device).to(dtype=hidden_states.dtype)
        else:
            self_mask_value = self.self_mask_value_float32.to(hidden_states.device).to(dtype=hidden_states.dtype)
        attn_weights = attn_weights.masked_fill(self_mask, self_mask_value)

        # FIX: dtype-aware softmax with stable logsumexp for precision alignment
        logits = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - logits)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, sorted_v_adj)

        # Merge chunks back
        attn_output = attn_output.reshape(bsz, self.num_attention_heads, -1, self.attention_head_size)

        # Unsort back to original order
        rev_sorted_indices = torch.argsort(sorted_indices, dim=-1)
        rev_sorted_indices_expanded = rev_sorted_indices.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        attn_output = torch.gather(attn_output, 2, rev_sorted_indices_expanded)

        # Average over hashes
        attn_output = attn_output.reshape(bsz, self.num_attention_heads, num_hashes, seq_len, self.attention_head_size)
        attn_output = attn_output.mean(dim=2)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.all_head_size).contiguous()

        return LSHSelfAttentionOutput(hidden_states=attn_output, attention_probs=None, buckets=buckets)

    def _standard_attention(self, query_key_vectors, value_vectors, attention_mask, bsz, seq_len):
        """Standard attention for short sequences."""
        attn_weights = torch.matmul(query_key_vectors, query_key_vectors.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.attention_head_size)

        # FIX: dtype-aware causal mask for precision alignment
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=query_key_vectors.device), diagonal=1
        )
        if query_key_vectors.dtype == torch.float16:
            mask_val = torch.tensor(-1e4, device=query_key_vectors.device, dtype=query_key_vectors.dtype)
        else:
            mask_val = torch.tensor(-1e9, device=query_key_vectors.device, dtype=query_key_vectors.dtype)
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), mask_val)

        # FIX: dtype-aware softmax with stable logsumexp for precision alignment
        logits = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - logits)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_vectors)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.all_head_size).contiguous()

        return LSHSelfAttentionOutput(hidden_states=attn_output, attention_probs=None, buckets=None)

    def _hash_vectors(self, vectors, num_hashes, attention_mask):
        """Hash vectors using random rotations with torch.manual_seed(hash_seed)"""
        batch_size = vectors.shape[0]

        if isinstance(self.num_buckets, int):
            rotation_size = self.num_buckets
            num_buckets = self.num_buckets
        else:
            rotation_size, num_buckets = 0, 1
            for bucket_factor in self.num_buckets:
                rotation_size += bucket_factor
                num_buckets *= bucket_factor

        vectors = vectors.detach()

        # Seed random generator for reproducible rotations
        if self.hash_seed is not None:
            torch.manual_seed(self.hash_seed)

        rotations_shape = (self.num_attention_heads, vectors.shape[-1], num_hashes, rotation_size // 2)
        random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)

        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
            buckets = torch.argmax(rotated_vectors, dim=-1)
        else:
            buckets, cur_sum, cur_product = None, 0, 1
            for bucket_factor in self.num_buckets:
                rv_factor = rotated_vectors[..., cur_sum:cur_sum + bucket_factor // 2]
                cur_sum += bucket_factor // 2
                rv_factor = torch.cat([rv_factor, -rv_factor], dim=-1)
                if buckets is None:
                    buckets = torch.argmax(rv_factor, dim=-1)
                else:
                    buckets = buckets + cur_product * torch.argmax(rv_factor, dim=-1)
                cur_product *= bucket_factor

        # Offset buckets for different hash rounds
        offsets = torch.arange(num_hashes, device=vectors.device)
        offsets = (offsets * num_buckets).view(1, 1, -1, 1)
        offsets = offsets.expand(batch_size, self.num_attention_heads, -1, -1)
        offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)

        return offset_buckets

    def _len_and_dim_norm(self, vectors):
        vectors = self._len_norm(vectors)
        vectors = vectors / math.sqrt(self.attention_head_size)
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        variance = torch.mean(x ** 2, -1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + epsilon)
        return norm_x

    @staticmethod
    def _look_adjacent(x, num_chunks_before, num_chunks_after):
        if num_chunks_before == 0 and num_chunks_after == 0:
            return x
        chunks = []
        if num_chunks_before > 0:
            chunks.append(x[:, :, :-1])  # All but last
        chunks.append(x)
        if num_chunks_after > 0:
            chunks.append(x[:, :, 1:])  # All but first
        # Need to handle properly by padding
        # Actually the adjacent lookup concatenates along the token dimension within each chunk
        adjacent_chunks = []
        num_chunks = x.shape[2]
        for chunk_idx in range(num_chunks):
            parts = []
            for offset in range(-num_chunks_before, num_chunks_after + 1):
                adj_idx = chunk_idx + offset
                if 0 <= adj_idx < num_chunks:
                    parts.append(x[:, :, adj_idx])
                else:
                    parts.append(torch.zeros_like(x[:, :, 0]))
            adjacent_chunks.append(torch.cat(parts, dim=-2))
        return torch.stack(adjacent_chunks, dim=2)


class LocalSelfAttention(nn.Module):
    """Local Self-Attention: key scaling K/sqrt(head_dim), chunked with sliding window"""
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        self.chunk_length = config.local_attn_chunk_length
        self.num_chunks_before = config.local_num_chunks_before
        self.num_chunks_after = getattr(config, "local_num_chunks_after", 0)

        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        self.dropout = config.local_attention_probs_dropout_prob

        self.mask_value_float16 = torch.tensor(-1e4)
        self.mask_value_float32 = torch.tensor(-1e9)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        bsz, seq_len, _ = hidden_states.shape

        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        query_vectors = query_vectors.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key_vectors = key_vectors.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_vectors = value_vectors.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Key normalization: scale keys K/sqrt(head_dim) instead of queries
        key_vectors = key_vectors / math.sqrt(self.attention_head_size)

        do_standard = seq_len <= self.chunk_length

        if not do_standard:
            num_chunks = seq_len // self.chunk_length
            query_vectors = query_vectors.reshape(bsz, self.num_attention_heads, num_chunks, self.chunk_length, self.attention_head_size)
            key_vectors = key_vectors.reshape(bsz, self.num_attention_heads, num_chunks, self.chunk_length, self.attention_head_size)
            value_vectors = value_vectors.reshape(bsz, self.num_attention_heads, num_chunks, self.chunk_length, self.attention_head_size)

            key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)

        attn_weights = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # FIX: dtype-aware causal mask handling for precision alignment
        if not do_standard:
            # Create chunk-local causal mask
            q_len = self.chunk_length
            k_len = key_vectors.shape[-2]
            # For each chunk, queries can only attend to keys with lower or equal position
            q_indices = torch.arange(q_len, device=hidden_states.device)
            k_indices_parts = []
            for offset in range(-self.num_chunks_before, self.num_chunks_after + 1):
                k_indices_parts.append(torch.arange(self.chunk_length, device=hidden_states.device) + (offset * self.chunk_length))
            k_indices = torch.cat(k_indices_parts)
            causal_mask = q_indices.unsqueeze(1) < k_indices.unsqueeze(0)
            if hidden_states.dtype == torch.float16:
                mask_val = self.mask_value_float16.to(hidden_states.device).to(dtype=hidden_states.dtype)
            else:
                mask_val = self.mask_value_float32.to(hidden_states.device).to(dtype=hidden_states.dtype)
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), mask_val)
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device), diagonal=1
            )
            if hidden_states.dtype == torch.float16:
                mask_val = self.mask_value_float16.to(hidden_states.device).to(dtype=hidden_states.dtype)
            else:
                mask_val = self.mask_value_float32.to(hidden_states.device).to(dtype=hidden_states.dtype)
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), mask_val)

        # Use logsumexp-based softmax for numerical stability
        logits = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_probs = torch.exp(attn_weights - logits)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_vectors)

        if not do_standard:
            attn_output = attn_output.reshape(bsz, self.num_attention_heads, seq_len, self.attention_head_size)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.all_head_size).contiguous()

        return LocalSelfAttentionOutput(hidden_states=attn_output, attention_probs=None)

    @staticmethod
    def _look_adjacent(x, num_chunks_before, num_chunks_after):
        if num_chunks_before == 0 and num_chunks_after == 0:
            return x
        num_chunks = x.shape[2]
        adjacent_chunks = []
        for chunk_idx in range(num_chunks):
            parts = []
            for offset in range(-num_chunks_before, num_chunks_after + 1):
                adj_idx = chunk_idx + offset
                if 0 <= adj_idx < num_chunks:
                    parts.append(x[:, :, adj_idx])
                else:
                    parts.append(torch.zeros_like(x[:, :, 0]))
            adjacent_chunks.append(torch.cat(parts, dim=-2))
        return torch.stack(adjacent_chunks, dim=2)


class ReformerAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        attn_type = config.attn_layers[layer_idx]
        if attn_type == "lsh":
            self.self_attention = LSHSelfAttention(config, layer_idx=layer_idx)
        elif attn_type == "local":
            self.self_attention = LocalSelfAttention(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

    def forward(self, hidden_states, attention_mask=None, num_hashes=None, buckets=None,
                output_attentions=False, orig_sequence_length=None):
        hidden_states = self.layer_norm(hidden_states)
        attn_outputs = self.self_attention(hidden_states, attention_mask=attention_mask)
        attn_output = self.output(attn_outputs.hidden_states)
        return attn_outputs._replace(hidden_states=attn_output)


class ChunkReformerFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.feed_forward_size)
        self.output = nn.Linear(config.feed_forward_size, config.hidden_size)
        self.dropout = config.hidden_dropout_prob

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.output(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class ReformerLayer(nn.Module):
    """ReformerLayer with RevNet: Y1=X1+f(X2), Y2=X2+g(Y1). Separate seeds for attention and FF dropout."""
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.attention = ReformerAttention(config, layer_idx)
        self.feed_forward = ChunkReformerFeedForward(config)
        self.attention_seed = None
        self.feed_forward_seed = None

    def _init_attention_seed(self):
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            self.attention_seed = int(torch.seed() % sys.maxsize)
        torch.manual_seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            device_idx = torch.cuda.current_device()
            self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            self.feed_forward_seed = int(torch.seed() % sys.maxsize)
        torch.manual_seed(self.feed_forward_seed)

    def forward(self, prev_attn_output, hidden_states, attention_mask=None, num_hashes=None,
                output_attentions=False, orig_sequence_length=None):
        with torch.no_grad():
            if self.training:
                self._init_attention_seed()

            attn_outputs = self.attention(
                hidden_states, attention_mask=attention_mask,
                num_hashes=num_hashes, output_attentions=output_attentions,
                orig_sequence_length=orig_sequence_length,
            )
            attn_output = attn_outputs.hidden_states

            # RevNet: Y_1 = X_1 + f(X_2)
            attn_output = prev_attn_output + attn_output

            if self.training:
                self._init_feed_forward_seed()

            # RevNet: Y_2 = X_2 + g(Y_1)
            hidden_states = hidden_states + self.feed_forward(attn_output)

        return ReformerOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            attention_probs=attn_outputs.attention_probs,
            buckets=getattr(attn_outputs, 'buckets', None),
        )


class ReformerEncoder(nn.Module):
    """ReformerEncoder: duplicates input as cat[h,h], applies layer_norm over 2*hidden_size"""
    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        self.layers = nn.ModuleList([ReformerLayer(config, i) for i in range(config.num_hidden_layers)])
        # RevNet: layer norm over 2*hidden_size
        self.layer_norm = nn.LayerNorm(2 * config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, num_hashes=None, output_attentions=False):
        # RevNet: duplicate input
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)

        # Split into attn_output and hidden_states streams
        attn_output, hidden_states_stream = torch.chunk(hidden_states, 2, dim=-1)

        for layer in self.layers:
            layer_outputs = layer(
                prev_attn_output=attn_output,
                hidden_states=hidden_states_stream,
                attention_mask=attention_mask,
                num_hashes=num_hashes,
                output_attentions=output_attentions,
            )
            attn_output = layer_outputs.attn_output
            hidden_states_stream = layer_outputs.hidden_states

        # Concatenate RevNet streams
        hidden_states = torch.cat([attn_output, hidden_states_stream], dim=-1)

        # Layer norm over 2*hidden_size
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        return hidden_states


class ReformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = ReformerEmbeddings(config)
        self.encoder = ReformerEncoder(config)

    def forward(self, input_ids, attention_mask=None, num_hashes=None):
        hidden_states = self.embeddings(input_ids)

        # Pad to chunk length if needed
        min_chunk = min(
            getattr(self.config, "lsh_attn_chunk_length", 64),
            getattr(self.config, "local_attn_chunk_length", 64)
        )
        seq_len = hidden_states.shape[1]
        padding_len = (min_chunk - seq_len % min_chunk) % min_chunk
        if padding_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding_len))
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, padding_len), value=0)

        encoder_output = self.encoder(hidden_states, attention_mask=attention_mask, num_hashes=num_hashes)

        # Remove padding
        if padding_len > 0:
            encoder_output = encoder_output[:, :seq_len]

        return encoder_output


class ReformerModelWithLMHead(nn.Module):
    """ReformerOnlyLMHead: nn.Linear(2*hidden_size, vocab_size) - takes concatenated RevNet output"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        config.is_decoder = True
        self.reformer = ReformerModel(config)
        # LM head: 2*hidden_size -> vocab_size (RevNet output is concatenated)
        self.lm_head_decoder = nn.Linear(2 * config.hidden_size, config.vocab_size, bias=False)
        self.lm_head_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.reformer(input_ids, attention_mask=attention_mask)
        logits = self.lm_head_decoder(hidden_states) + self.lm_head_bias
        return logits


# FIX: Static hardcoded ReformerConfig class replacing runtime config loading
class ReformerConfig:
    """Hardcoded config for google/reformer-enwik8 (bs1024, seq32)"""
    vocab_size = 258
    hidden_size = 256
    num_attention_heads = 2
    attention_head_size = 128
    num_hidden_layers = 6
    feed_forward_size = 512
    hidden_dropout_prob = 0.05
    layer_norm_eps = 1e-12
    # Axial position embeddings: factorized [64, 64] with dims [64, 192]
    axial_pos_shape = [64, 64]
    axial_pos_embds_dim = [64, 192]
    # LSH attention
    lsh_attn_chunk_length = 64
    num_hashes = 1
    num_buckets = 64
    lsh_num_chunks_before = 1
    lsh_num_chunks_after = 0
    lsh_attention_probs_dropout_prob = 0.0
    # Local attention
    local_attn_chunk_length = 64
    local_num_chunks_before = 1
    local_num_chunks_after = 0
    local_attention_probs_dropout_prob = 0.05
    # Attention layers pattern: alternating LSH and local
    attn_layers = ["lsh", "local", "lsh", "local", "lsh", "local"]
    is_decoder = True


# FIX: Updated Model interface - __init__(self, config), get_init_inputs() returns [config]
class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ReformerModelWithLMHead(config)

    def forward(self, x):
        return self.model(x)


model_name = "google/reformer-enwik8"
config = ReformerConfig()
vocab_size = config.vocab_size
sequence_length = 32
batch_size = 1024


def get_inputs():
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]


def get_init_inputs():
    # FIX: Return config for static initialization
    return [ReformerConfig()]
