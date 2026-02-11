"""Self-contained Reformer benchmark: google/reformer-enwik8, batch_size=32, sequence_length=256"""

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
    """Factorized 2D position embeddings matching HF structure.
    Parameters have 3D shapes for broadcasting: (shape[axis], 1, dim) per axis."""
    def __init__(self, config):
        super().__init__()
        self.axial_pos_shape = config.axial_pos_shape  # e.g. [64, 64]
        self.axial_pos_embds_dim = config.axial_pos_embds_dim  # e.g. [64, 192]
        self.dropout = config.hidden_dropout_prob

        # Create parameter lists matching HF's 3D shape: (shape[axis], 1, dim) per axis
        self.weights = nn.ParameterList()
        for axis_idx in range(len(self.axial_pos_shape)):
            ax_shape = [1] * len(self.axial_pos_shape)
            ax_shape[axis_idx] = self.axial_pos_shape[axis_idx]
            ax_shape = tuple(ax_shape) + (self.axial_pos_embds_dim[axis_idx],)
            self.weights.append(nn.Parameter(torch.ones(ax_shape, dtype=torch.float32)))

    def forward(self, position_ids):
        batch_size = position_ids.shape[0]
        seq_len = position_ids.shape[1]

        # Lazy broadcast: expand doesn't allocate memory
        broadcasted_weights = [
            weight.expand((batch_size,) + tuple(self.axial_pos_shape) + weight.shape[-1:])
            for weight in self.weights
        ]

        if not self.training:
            # Eval mode: only materialize needed rows for memory efficiency (matches HF)
            max_position_id = position_ids.max().item()
            required_rows = -(-(max_position_id + 1) // self.axial_pos_shape[1])

            # Slice to needed rows only along the first position axis
            position_encodings = torch.cat(
                [weight[:, :required_rows] for weight in broadcasted_weights], dim=-1
            )
            position_encodings = position_encodings.reshape(batch_size, -1, position_encodings.shape[-1])

            # Select exact positions for each batch item
            position_encodings = torch.cat(
                [
                    torch.index_select(position_encodings[i], 0, position_ids[i]).unsqueeze(0)
                    for i in range(batch_size)
                ],
                dim=0,
            )
        else:
            # Training mode: full position grid with dropout2d
            full_len = 1
            for s in self.axial_pos_shape:
                full_len *= s

            if self.dropout > 0:
                weights = torch.cat(broadcasted_weights, dim=-1)
                transposed_weights = weights.transpose(2, 1)
                dropped_transposed_weights = F.dropout2d(
                    transposed_weights, p=self.dropout, training=True
                )
                dropped_weights = dropped_transposed_weights.transpose(2, 1)
                position_encodings = dropped_weights.reshape(batch_size, full_len, -1)
            else:
                position_encodings = torch.cat(
                    [w.reshape(batch_size, full_len, -1) for w in broadcasted_weights],
                    dim=-1,
                )
            position_encodings = position_encodings[:, :seq_len, :]

        return position_encodings


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
    """LSH Self-Attention matching HF's implementation:
    - Hash RAW vectors (not normalized)
    - Normalize only keys (not queries)
    - Position-based self-mask (prevent self-attention)
    - Causal mask for decoder
    - Circular wrapping in _look_adjacent
    - Logsumexp-weighted hash combination
    """
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_decoder = getattr(config, 'is_decoder', False)

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        self.chunk_length = config.lsh_attn_chunk_length
        self.num_hashes = config.num_hashes
        self.num_buckets = config.num_buckets
        self.num_chunks_before = getattr(config, "lsh_num_chunks_before", 1)
        self.num_chunks_after = getattr(config, "lsh_num_chunks_after", 0)

        # Match HF: use config.hash_seed (default None)
        self.hash_seed = getattr(config, 'hash_seed', None)

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
        query_key_vectors = query_key_vectors.transpose(1, 2)  # (B, H, S, D)
        value_vectors = value_vectors.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)
        value_vectors = value_vectors.transpose(1, 2)  # (B, H, S, D)

        # #region agent log
        import json as _json_lsh
        _log_path_lsh = "/home/xiexinch/KernelBench/.cursor/debug.log"
        _rng_cpu = torch.random.get_rng_state()[:8].tolist()
        _rng_cuda = torch.cuda.get_rng_state()[:8].tolist() if torch.cuda.is_available() else []
        with open(_log_path_lsh, "a") as _lf_lsh:
            _lf_lsh.write(_json_lsh.dumps({"hypothesisId": "H-LSH-rng", "location": f"LSH.forward:layer{self.layer_idx}", "message": "rng_state_before_hash", "data": {"layer_idx": self.layer_idx, "rng_cpu_first8": _rng_cpu, "rng_cuda_first8": _rng_cuda, "qk_first3": query_key_vectors[0, 0, 0, :3].float().cpu().tolist(), "v_first3": value_vectors[0, 0, 0, :3].float().cpu().tolist(), "seq_len": seq_len}}) + "\n")
        # #endregion

        # For short sequences, use standard attention
        do_standard = seq_len <= self.chunk_length
        if do_standard:
            return self._standard_attention(query_key_vectors, value_vectors, attention_mask, bsz, seq_len)

        # Hash RAW (unnormalized) vectors - match HF
        if buckets is None:
            buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)

        # #region agent log
        with open(_log_path_lsh, "a") as _lf_lsh:
            _lf_lsh.write(_json_lsh.dumps({"hypothesisId": "H-LSH-buckets", "location": f"LSH.forward:layer{self.layer_idx}", "message": "buckets_after_hash", "data": {"layer_idx": self.layer_idx, "buckets_shape": list(buckets.shape), "buckets_first10": buckets[0, 0, :10].cpu().tolist(), "buckets_unique": len(buckets[0, 0].unique().cpu().tolist())}}) + "\n")
        # #endregion

        # Sort by buckets - get sorting indices
        sorted_bucket_idx = self._stable_argsort(buckets)
        # Position indices for tracking original positions (match HF: sorted_bucket_idx % seq_len)
        sorted_bucket_idx_per_hash = sorted_bucket_idx % seq_len

        # Create undo sort indices
        indices = torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device).view(1, 1, -1).expand_as(sorted_bucket_idx)
        undo_sorted_bucket_idx = sorted_bucket_idx.new_zeros(sorted_bucket_idx.shape)
        undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

        # Gather vectors by sorted position indices (match HF's _gather_by_expansion)
        expanded_idx = sorted_bucket_idx_per_hash.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        qk_repeated = query_key_vectors.repeat(1, 1, num_hashes, 1)
        v_repeated = value_vectors.repeat(1, 1, num_hashes, 1)
        sorted_qk = torch.gather(qk_repeated, 2, expanded_idx)
        sorted_v = torch.gather(v_repeated, 2, expanded_idx)

        # Chunk into groups
        total_len = sorted_qk.shape[2]
        chunk_len = self.chunk_length
        num_chunks = total_len // chunk_len

        sorted_qk = sorted_qk.reshape(bsz, self.num_attention_heads, num_chunks, chunk_len, self.attention_head_size)
        sorted_v = sorted_v.reshape(bsz, self.num_attention_heads, num_chunks, chunk_len, self.attention_head_size)

        # Normalize ONLY keys, queries stay raw (match HF)
        key_vectors = self._len_and_dim_norm(sorted_qk)
        query_vectors = sorted_qk

        # Look adjacent with circular wrapping (match HF)
        key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
        sorted_v = self._look_adjacent(sorted_v, self.num_chunks_before, self.num_chunks_after)

        # Compute attention scores
        attn_weights = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # Position-based indices for masking (match HF)
        query_bucket_idx = sorted_bucket_idx_per_hash.reshape(bsz, self.num_attention_heads, num_chunks, chunk_len)
        key_value_bucket_idx = self._look_adjacent(query_bucket_idx, self.num_chunks_before, self.num_chunks_after)

        # Get correct mask values depending on precision
        if attn_weights.dtype == torch.float16:
            self_mask_value = self.self_mask_value_float16.half()
            mask_value = self.mask_value_float16.half()
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32

        # Causal + attention mask (match HF's _compute_attn_mask)
        mask = self._compute_attn_mask(
            query_bucket_idx, key_value_bucket_idx, attention_mask,
            attn_weights.shape, do_standard_self_attention=False
        )
        if mask is not None:
            attn_weights = torch.where(mask, attn_weights, mask_value)

        # Self-mask: prevent token from attending to itself (POSITION-based, match HF)
        self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
            query_bucket_idx.device
        )
        attn_weights = torch.where(self_mask, attn_weights, self_mask_value)

        # Softmax via logsumexp
        logits = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_probs = torch.exp(attn_weights - logits)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Attend values
        attn_output = torch.matmul(attn_probs, sorted_v)

        # Merge chunks: flatten chunk dims
        logits = logits.flatten(start_dim=2, end_dim=3).squeeze(-1)  # (B, H, num_hashes * S)
        attn_output = attn_output.flatten(start_dim=2, end_dim=3)  # (B, H, num_hashes * S, D)

        # Unsort back to original order
        undo_expanded = undo_sorted_bucket_idx.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        attn_output = torch.gather(attn_output, 2, undo_expanded)
        logits = torch.gather(logits, 2, undo_sorted_bucket_idx)

        # Combine hashes with logsumexp weighting (match HF, not simple mean)
        if num_hashes > 1:
            attn_output = attn_output.reshape(bsz, self.num_attention_heads, num_hashes, seq_len, self.attention_head_size)
            logits = logits.reshape(bsz, self.num_attention_heads, num_hashes, seq_len).unsqueeze(-1)
            probs_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
            attn_output = torch.sum(attn_output * probs_vectors, dim=2)
        else:
            attn_output = attn_output.reshape(bsz, self.num_attention_heads, seq_len, self.attention_head_size)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.all_head_size).contiguous()

        return LSHSelfAttentionOutput(hidden_states=attn_output, attention_probs=None, buckets=buckets)

    def _standard_attention(self, query_key_vectors, value_vectors, attention_mask, bsz, seq_len):
        """Standard attention for short sequences (matches HF)."""
        # In standard mode, use position indices directly
        sorted_bucket_idx_per_hash = torch.arange(seq_len, device=query_key_vectors.device).repeat(
            bsz, self.num_attention_heads, 1
        )
        # Normalize only keys
        key_vectors = self._len_and_dim_norm(query_key_vectors)
        # Queries stay raw
        attn_weights = torch.matmul(query_key_vectors, key_vectors.transpose(-1, -2))

        # Mask values
        if attn_weights.dtype == torch.float16:
            self_mask_value = self.self_mask_value_float16.half()
            mask_value = self.mask_value_float16.half()
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32

        # Causal + attention mask
        mask = self._compute_attn_mask(
            sorted_bucket_idx_per_hash, sorted_bucket_idx_per_hash, attention_mask,
            attn_weights.shape, do_standard_self_attention=True
        )
        if mask is not None:
            attn_weights = torch.where(mask, attn_weights, mask_value)

        # Self-mask
        self_mask = torch.ne(
            sorted_bucket_idx_per_hash.unsqueeze(-1),
            sorted_bucket_idx_per_hash.unsqueeze(-2)
        ).to(query_key_vectors.device)
        attn_weights = torch.where(self_mask, attn_weights, self_mask_value)

        logits = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_probs = torch.exp(attn_weights - logits)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_vectors)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.all_head_size).contiguous()

        return LSHSelfAttentionOutput(hidden_states=attn_output, attention_probs=None, buckets=None)

    def _compute_attn_mask(self, query_indices, key_indices, attention_mask, query_key_dot_shape, do_standard_self_attention):
        """Compute attention mask including causal mask (matches HF)."""
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)[:, None, :]
            if not do_standard_self_attention:
                attention_mask = attention_mask[:, None, :]
                attention_mask = attention_mask.expand(query_indices.shape[:-1] + (-1,))
                attention_mask = torch.gather(attention_mask, -1, key_indices)
            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dot_shape)

        # Causal mask for decoder
        if self.is_decoder:
            causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask

        return attention_mask

    def _hash_vectors(self, vectors, num_hashes, attention_mask):
        """Hash vectors using random rotations (matches HF)."""
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

        # Match HF: only seed if hash_seed is not None
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
    def _stable_argsort(vector):
        """Stable argsort matching HF's implementation."""
        scale_offset = torch.arange(vector.shape[-1], device=vector.device).view(1, 1, -1).expand_as(vector)
        scaled_vector = vector.shape[-1] * vector + (scale_offset % vector.shape[-1])
        return scaled_vector.argsort(dim=-1)

    @staticmethod
    def _look_adjacent(x, num_chunks_before, num_chunks_after):
        """Circular wrapping adjacent chunk lookup (matches HF)."""
        if num_chunks_before == 0 and num_chunks_after == 0:
            return x
        slices = []
        for i in range(-num_chunks_before, num_chunks_after + 1):
            if i == 0:
                slices.append(x)
            else:
                slices.append(torch.cat([x[:, :, i:, ...], x[:, :, :i, ...]], dim=2))
        return torch.cat(slices, dim=3)


class LocalSelfAttention(nn.Module):
    """Local Self-Attention: key scaling K/sqrt(head_dim), chunked with sliding window"""
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size
        self.is_decoder = getattr(config, 'is_decoder', False)

        self.chunk_length = config.local_attn_chunk_length
        self.num_chunks_before = config.local_num_chunks_before
        self.num_chunks_after = getattr(config, "local_num_chunks_after", 0)

        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)

        self.dropout = config.local_attention_probs_dropout_prob

        self.mask_value_float16 = torch.tensor(-1e4)
        self.mask_value_float32 = torch.tensor(-1e9)

    @staticmethod
    def _look_adjacent(x, num_chunks_before, num_chunks_after):
        """Circular wrapping adjacent chunk lookup (matches HF)."""
        if num_chunks_before == 0 and num_chunks_after == 0:
            return x
        slices = []
        for i in range(-num_chunks_before, num_chunks_after + 1):
            if i == 0:
                slices.append(x)
            else:
                slices.append(torch.cat([x[:, :, i:, ...], x[:, :, :i, ...]], dim=2))
        return torch.cat(slices, dim=3)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        bsz, seq_len, _ = hidden_states.shape

        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        query_vectors = query_vectors.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key_vectors = key_vectors.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_vectors = value_vectors.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Key normalization: scale keys K/sqrt(head_dim) instead of queries (matches HF)
        key_vectors = key_vectors / math.sqrt(self.attention_head_size)

        # Absolute position indices for masking (matches HF)
        indices = torch.arange(seq_len, device=hidden_states.device).repeat(bsz, self.num_attention_heads, 1)

        do_standard = seq_len <= self.chunk_length

        if not do_standard:
            num_chunks = seq_len // self.chunk_length
            query_vectors = query_vectors.reshape(bsz, self.num_attention_heads, num_chunks, self.chunk_length, self.attention_head_size)
            key_vectors = key_vectors.reshape(bsz, self.num_attention_heads, num_chunks, self.chunk_length, self.attention_head_size)
            value_vectors = value_vectors.reshape(bsz, self.num_attention_heads, num_chunks, self.chunk_length, self.attention_head_size)

            # Chunk indices for masking
            query_indices = indices.reshape(bsz, self.num_attention_heads, num_chunks, self.chunk_length)
            key_indices = indices.reshape(bsz, self.num_attention_heads, num_chunks, self.chunk_length)

            # Look adjacent for keys, values, and key_indices (circular wrapping)
            key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)
            key_indices = self._look_adjacent(key_indices, self.num_chunks_before, self.num_chunks_after)
        else:
            query_indices = key_indices = indices

        attn_weights = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # Compute attention mask using HF's _compute_attn_mask approach
        mask = self._compute_attn_mask(query_indices, key_indices, attention_mask, attn_weights.shape, do_standard)

        if mask is not None:
            if attn_weights.dtype == torch.float16:
                mask_val = self.mask_value_float16.to(hidden_states.device).half()
            else:
                mask_val = self.mask_value_float32.to(hidden_states.device)
            attn_weights = torch.where(mask, attn_weights, mask_val)

        # Use logsumexp-based softmax for numerical stability (matches HF)
        logits = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_probs = torch.exp(attn_weights - logits)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_vectors)

        if not do_standard:
            attn_output = attn_output.flatten(start_dim=2, end_dim=3)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.all_head_size).contiguous()

        return LocalSelfAttentionOutput(hidden_states=attn_output, attention_probs=None)

    def _compute_attn_mask(self, query_indices, key_indices, attention_mask, query_key_dots_shape, do_standard):
        """Compute attention mask including causal mask (matches HF LocalSelfAttention._compute_attn_mask)."""
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)[:, None, :]
            if not do_standard:
                attention_mask = attention_mask.reshape(attention_mask.shape[0], 1, -1, self.chunk_length)
                attention_mask = self._look_adjacent(attention_mask, self.num_chunks_before, self.num_chunks_after)
            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dots_shape)

        # Causal mask for decoder (matches HF: only when is_decoder=True)
        if self.is_decoder is True:
            causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask

        return attention_mask


class ReformerSelfOutput(nn.Module):
    """Wrapper for attention output linear (matches HF key: attention.output.dense.weight)."""
    def __init__(self, config):
        super().__init__()
        all_head_size = config.num_attention_heads * config.attention_head_size
        self.dense = nn.Linear(all_head_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.dense(hidden_states)


class ReformerAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output = ReformerSelfOutput(config)

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


class ReformerFeedForwardDense(nn.Module):
    """Wrapper for FF input linear (matches HF key: feed_forward.dense.dense.weight)."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.feed_forward_size)

    def forward(self, hidden_states):
        return self.dense(hidden_states)


class ReformerFeedForwardOutput(nn.Module):
    """Wrapper for FF output linear (matches HF key: feed_forward.output.dense.weight)."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.feed_forward_size, config.hidden_size)

    def forward(self, hidden_states):
        return self.dense(hidden_states)


class ChunkReformerFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = ReformerFeedForwardDense(config)
        self.output = ReformerFeedForwardOutput(config)
        self.dropout = config.hidden_dropout_prob

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        # Match HF: linear → dropout → activation (config.hidden_act defaults to "relu")
        hidden_states = self.dense(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = F.relu(hidden_states)
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


class ReformerOnlyLMHead(nn.Module):
    """LM head matching HF structure: lm_head.decoder.weight, lm_head.bias.
    HF does NOT use bias in forward (decoder has bias=False, self.bias exists but is unused)."""
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Linear(2 * config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        return self.decoder(hidden_states)


class ReformerModelWithLMHead(nn.Module):
    """Reformer with LM head (matches HF ReformerModelWithLMHead structure)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        config.is_decoder = True
        self.reformer = ReformerModel(config)
        # LM head: 2*hidden_size -> vocab_size (RevNet output is concatenated)
        self.lm_head = ReformerOnlyLMHead(config)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.reformer(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return logits


# FIX: Static hardcoded ReformerConfig class replacing runtime config loading
class ReformerConfig:
    """Hardcoded config for google/reformer-enwik8 (bs32, seq256)"""
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
sequence_length = 256
batch_size = 32


def get_inputs():
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]


def get_init_inputs():
    # FIX: Return config for static initialization
    return [ReformerConfig()]
