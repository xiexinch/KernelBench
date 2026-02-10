"""Self-contained ELECTRA benchmark: batch_size=1, sequence_length=511"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ElectraEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size,
                                            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids, token_type_ids=None):
        bsz, seq_len = input_ids.size()
        position_ids = self.position_ids[:, :seq_len]

        if token_type_ids is None:
            token_type_ids = torch.zeros(bsz, seq_len, dtype=torch.long, device=input_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ElectraSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.size()

        query_layer = self.query(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key_layer = self.key(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = self.value(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        attn_weights = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_weights = attn_weights * (self.attention_head_size ** -0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_layer)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()

        return attn_output


class ElectraSelfOutput(nn.Module):
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


class ElectraAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = ElectraSelfAttention(config)
        self.output = ElectraSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class ElectraIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class ElectraOutput(nn.Module):
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


class ElectraLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ElectraAttention(config)
        self.intermediate = ElectraIntermediate(config)
        self.output = ElectraOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class ElectraEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([ElectraLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class ElectraModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = ElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
        else:
            self.embeddings_project = None

        self.encoder = ElectraEncoder(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids)

        if self.embeddings_project is not None:
            embedding_output = self.embeddings_project(embedding_output)

        # Create extended attention mask for encoder
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(embedding_output.dtype).min
        else:
            extended_attention_mask = None

        encoder_output = self.encoder(embedding_output, attention_mask=extended_attention_mask)
        return encoder_output


class ElectraGeneratorPredictions(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class ElectraForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.electra = ElectraModel(config)
        self.generator_predictions = ElectraGeneratorPredictions(config)
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        # Weight tying
        self.generator_lm_head.weight = self.electra.embeddings.word_embeddings.weight

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        hidden_states = self.electra(input_ids, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
        prediction_scores = self.generator_predictions(hidden_states)
        logits = self.generator_lm_head(prediction_scores)
        return logits


class ElectraConfig:
    def __init__(self):
        # google/electra-small-discriminator configuration
        self.vocab_size = 30522
        self.embedding_size = 128
        self.hidden_size = 256
        self.num_hidden_layers = 12
        self.num_attention_heads = 4
        self.intermediate_size = 1024
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.layer_norm_eps = 1e-12
        self.pad_token_id = 0


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = ElectraConfig()
        self.model = ElectraForCausalLM(config)

    def forward(self, x):
        return self.model(x)


# Benchmark configuration
vocab_size = 30522
sequence_length = 511
batch_size = 1


def get_inputs():
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]


def get_init_inputs():
    return []
