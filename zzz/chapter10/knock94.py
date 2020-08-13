import math

import torch
from torch import nn
import numpy as np
from collections import defaultdict

PAD_TOKEN_ID = 0
PAD_TOKEN_INDEX = 0


class Beam:

    def __init__(self, beam_size=8, min_length=0, n_top=1, ranker=None,
                 start_token_id=2, end_token_id=3):
        self.beam_size = beam_size
        self.min_length = min_length
        self.ranker = ranker

        self.end_token_id = end_token_id
        self.top_sentence_ended = False

        self.prev_ks = []
        self.next_ys = [torch.LongTensor(beam_size).fill_(start_token_id)]  # remove padding

        self.current_scores = torch.FloatTensor(beam_size).zero_()
        self.all_scores = []

        # The attentions (matrix) for each time.
        self.all_attentions = []

        self.finished = []

        # Time and k pair for finished.
        self.finished = []
        self.n_top = n_top

        self.ranker = ranker

    def advance(self, next_log_probs, current_attention):
        # next_probs : beam_size X vocab_size
        # current_attention: (target_seq_len=1, beam_size, source_seq_len)

        vocabulary_size = next_log_probs.size(1)
        # current_beam_size = next_log_probs.size(0)

        current_length = len(self.next_ys)
        if current_length < self.min_length:
            for beam_index in range(len(next_log_probs)):
                next_log_probs[beam_index][self.end_token_id] = -1e10

        if len(self.prev_ks) > 0:
            beam_scores = next_log_probs + self.current_scores.unsqueeze(1).expand_as(next_log_probs)
            # Don't let EOS have children.
            last_y = self.next_ys[-1]
            for beam_index in range(last_y.size(0)):
                if last_y[beam_index] == self.end_token_id:
                    beam_scores[beam_index] = -1e10  # -1e20 raises error when executing
        else:
            beam_scores = next_log_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        top_scores, top_score_ids = flat_beam_scores.topk(k=self.beam_size, dim=0, largest=True, sorted=True)

        self.current_scores = top_scores
        self.all_scores.append(self.current_scores)

        prev_k = top_score_ids / vocabulary_size  # (beam_size, )
        next_y = top_score_ids - prev_k * vocabulary_size  # (beam_size, )

        self.prev_ks.append(prev_k)
        self.next_ys.append(next_y)
        # for RNN, dim=1 and for transformer, dim=0.
        prev_attention = current_attention.index_select(dim=0,
                                                        index=prev_k)  # (target_seq_len=1, beam_size, source_seq_len)
        self.all_attentions.append(prev_attention)

        for beam_index, last_token_id in enumerate(next_y):
            if last_token_id == self.end_token_id:
                # skip scoring
                self.finished.append((self.current_scores[beam_index], len(self.next_ys) - 1, beam_index))

        if next_y[0] == self.end_token_id:
            self.top_sentence_ended = True

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def done(self):
        return self.top_sentence_ended and len(self.finished) >= self.n_top

    def get_hypothesis(self, timestep, k):
        hypothesis, attentions = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hypothesis.append(self.next_ys[j + 1][k])
            # for RNN, [:, k, :], and for trnasformer, [k, :, :]
            attentions.append(self.all_attentions[j][k, :, :])
            k = self.prev_ks[j][k]
        attentions_tensor = torch.stack(attentions[::-1]).squeeze(1)  # (timestep, source_seq_len)
        return hypothesis[::-1], attentions_tensor

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                # global_scores = self.global_scorer.score(self, self.scores)
                # s = global_scores[i]
                s = self.current_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished = sorted(self.finished, key=lambda a: a[0], reverse=True)
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout_prob (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, num_embeddings, embedding_dim, dim, dropout_prob=0., padding_idx=0, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2) *
                              -(math.log(10000.0) / dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embbedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.weight = self.embbedding.weight
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        x = self.embbedding(x)
        x = x * math.sqrt(self.dim)
        if step is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x


def build_model(config, source_vocabulary_size, target_vocabulary_size):
    if config['positional_encoding']:
        source_embedding = PositionalEncoding(
            num_embeddings=source_vocabulary_size,
            embedding_dim=config['d_model'],
            dim=config['d_model'])  # why dim?
        target_embedding = PositionalEncoding(
            num_embeddings=target_vocabulary_size,
            embedding_dim=config['d_model'],
            dim=config['d_model'])  # why dim?
    else:
        source_embedding = nn.Embedding(
            num_embeddings=source_vocabulary_size,
            embedding_dim=config['d_model'])
        target_embedding = nn.Embedding(
            num_embeddings=target_vocabulary_size,
            embedding_dim=config['d_model'])

    encoder = TransformerEncoder(
        layers_count=config['layers_count'],
        d_model=config['d_model'],
        heads_count=config['heads_count'],
        d_ff=config['d_ff'],
        dropout_prob=config['dropout_prob'],
        embedding=source_embedding)

    decoder = TransformerDecoder(
        layers_count=config['layers_count'],
        d_model=config['d_model'],
        heads_count=config['heads_count'],
        d_ff=config['d_ff'],
        dropout_prob=config['dropout_prob'],
        embedding=target_embedding)

    model = Transformer(encoder, decoder)

    return model


class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sources, inputs):
        # sources : (batch_size, sources_len)
        # inputs : (batch_size, targets_len - 1)
        batch_size, sources_len = sources.size()
        batch_size, inputs_len = inputs.size()

        sources_mask = pad_masking(sources, sources_len)
        memory_mask = pad_masking(sources, inputs_len)
        inputs_mask = subsequent_masking(inputs) | pad_masking(inputs, inputs_len)

        memory = self.encoder(sources, sources_mask)  # (batch_size, seq_len, d_model)
        outputs, state = self.decoder(inputs, memory, memory_mask, inputs_mask)  # (batch_size, seq_len, d_model)
        return outputs


class TransformerEncoder(nn.Module):

    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob, embedding):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.embedding = embedding
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)]
        )

    def forward(self, sources, mask):
        """

        args:
           sources: embedded_sequence, (batch_size, seq_len, embed_size)
        """
        sources = self.embedding(sources)

        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(sources, mask)

        return sources


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention_layer = Sublayer(MultiHeadAttention(heads_count, d_model, dropout_prob), d_model)
        self.pointwise_feedforward_layer = Sublayer(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, sources, sources_mask):
        # x: (batch_size, seq_len, d_model)

        sources = self.self_attention_layer(sources, sources, sources, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)

        return sources


class TransformerDecoder(nn.Module):

    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob, embedding):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.embedding = embedding
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)]
        )
        self.generator = nn.Linear(embedding.embedding_dim, embedding.num_embeddings)
        self.generator.weight = self.embedding.weight

    def forward(self, inputs, memory, memory_mask, inputs_mask=None, state=None):
        # inputs: (batch_size, seq_len - 1, d_model)
        # memory: (batch_size, seq_len, d_model)

        inputs = self.embedding(inputs)
        # if state is not None:
        #     inputs = torch.cat([state.previous_inputs, inputs], dim=1)
        #
        #     state.previous_inputs = inputs

        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            if state is None:
                inputs = decoder_layer(inputs, memory, memory_mask, inputs_mask)
            else:  # Use cache
                layer_cache = state.layer_caches[layer_index]
                # print('inputs_mask', inputs_mask)
                inputs = decoder_layer(inputs, memory, memory_mask, inputs_mask, layer_cache)

                state.update_state(
                    layer_index=layer_index,
                    layer_mode='self-attention',
                    key_projected=decoder_layer.self_attention_layer.sublayer.key_projected,
                    value_projected=decoder_layer.self_attention_layer.sublayer.value_projected,
                )
                state.update_state(
                    layer_index=layer_index,
                    layer_mode='memory-attention',
                    key_projected=decoder_layer.memory_attention_layer.sublayer.key_projected,
                    value_projected=decoder_layer.memory_attention_layer.sublayer.value_projected,
                )

        generated = self.generator(inputs)  # (batch_size, seq_len, vocab_size)
        return generated, state

    def init_decoder_state(self, **args):
        return DecoderState()


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention_layer = Sublayer(
            MultiHeadAttention(heads_count, d_model, dropout_prob, mode='self-attention'), d_model)
        self.memory_attention_layer = Sublayer(
            MultiHeadAttention(heads_count, d_model, dropout_prob, mode='memory-attention'), d_model)
        self.pointwise_feedforward_layer = Sublayer(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)

    def forward(self, inputs, memory, memory_mask, inputs_mask, layer_cache=None):
        # print('self attention')
        # print('inputs_mask', inputs_mask)
        inputs = self.self_attention_layer(inputs, inputs, inputs, inputs_mask, layer_cache)
        # print('memory attention')
        inputs = self.memory_attention_layer(inputs, memory, memory, memory_mask, layer_cache)
        inputs = self.pointwise_feedforward_layer(inputs)
        return inputs


class Sublayer(nn.Module):

    def __init__(self, sublayer, d_model):
        super(Sublayer, self).__init__()

        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)


class LayerNormalization(nn.Module):

    def __init__(self, features_count, epsilon=1e-6):
        super(LayerNormalization, self).__init__()

        self.gain = nn.Parameter(torch.ones(features_count))
        self.bias = nn.Parameter(torch.zeros(features_count))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class MultiHeadAttention(nn.Module):

    def __init__(self, heads_count, d_model, dropout_prob, mode='self-attention'):
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads_count == 0
        assert mode in ('self-attention', 'memory-attention')

        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.mode = mode
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)

        self.attention = None
        # For cache
        self.key_projected = None
        self.value_projected = None

    def forward(self, query, key, value, mask=None, layer_cache=None):
        """

        Args:
            query: (batch_size, query_len, model_dim)
            key: (batch_size, key_len, model_dim)
            value: (batch_size, value_len, model_dim)
            mask: (batch_size, query_len, key_len)
            state: DecoderState
        """
        # print('attention mask', mask)
        batch_size, query_len, d_model = query.size()

        d_head = d_model // self.heads_count

        query_projected = self.query_projection(query)
        # print('query_projected', query_projected.shape)
        if layer_cache is None or layer_cache[self.mode] is None:  # Don't use cache
            key_projected = self.key_projection(key)
            value_projected = self.value_projection(value)
        else:  # Use cache
            if self.mode == 'self-attention':
                key_projected = self.key_projection(key)
                value_projected = self.value_projection(value)

                key_projected = torch.cat([key_projected, layer_cache[self.mode]['key_projected']], dim=1)
                value_projected = torch.cat([value_projected, layer_cache[self.mode]['value_projected']], dim=1)
            elif self.mode == 'memory-attention':
                key_projected = layer_cache[self.mode]['key_projected']
                value_projected = layer_cache[self.mode]['value_projected']

        # For cache
        self.key_projected = key_projected
        self.value_projected = value_projected

        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()

        query_heads = query_projected.view(batch_size, query_len, self.heads_count, d_head).transpose(1,
                                                                                                      2)  # (batch_size, heads_count, query_len, d_head)
        # print('query_heads', query_heads.shape)
        # print(batch_size, key_len, self.heads_count, d_head)
        # print(key_projected.shape)
        key_heads = key_projected.view(batch_size, key_len, self.heads_count, d_head).transpose(1,
                                                                                                2)  # (batch_size, heads_count, key_len, d_head)
        value_heads = value_projected.view(batch_size, value_len, self.heads_count, d_head).transpose(1,
                                                                                                      2)  # (batch_size, heads_count, value_len, d_head)

        attention_weights = self.scaled_dot_product(query_heads,
                                                    key_heads)  # (batch_size, heads_count, query_len, key_len)

        if mask is not None:
            # print('mode', self.mode)
            # print('mask', mask.shape)
            # print('attention_weights', attention_weights.shape)
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(mask_expanded, -1e18)

        self.attention = self.softmax(attention_weights)  # Save attention to the object
        # print('attention_weights', attention_weights.shape)
        attention_dropped = self.dropout(self.attention)
        context_heads = torch.matmul(attention_dropped, value_heads)  # (batch_size, heads_count, query_len, d_head)
        # print('context_heads', context_heads.shape)
        context_sequence = context_heads.transpose(1, 2).contiguous()  # (batch_size, query_len, heads_count, d_head)
        context = context_sequence.view(batch_size, query_len, d_model)  # (batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        # print('final_output', final_output.shape)

        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        """

        Args:
             query_heads: (batch_size, heads_count, query_len, d_head)
             key_heads: (batch_size, heads_count, key_len, d_head)
        """
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)  # (batch_size, heads_count, query_len, key_len)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):

    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        """

        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)


class DecoderState:

    def __init__(self):
        self.previous_inputs = torch.tensor([])
        self.layer_caches = defaultdict(lambda: {'self-attention': None, 'memory-attention': None})

    def update_state(self, layer_index, layer_mode, key_projected, value_projected):
        self.layer_caches[layer_index][layer_mode] = {
            'key_projected': key_projected,
            'value_projected': value_projected
        }

    # def repeat_beam_size_times(self, beam_size): # memory만 repeat하면 되는데 state에 memory는 넣지 않기로 했다.
    #     self.
    #     self.src = self.src.data.repeat(beam_size, 1)

    def beam_update(self, positions):
        for layer_index in self.layer_caches:
            for mode in ('self-attention', 'memory-attention'):
                if self.layer_caches[layer_index][mode] is not None:
                    for projection in self.layer_caches[layer_index][mode]:
                        cache = self.layer_caches[layer_index][mode][projection]
                        if cache is not None:
                            cache.data.copy_(cache.data.index_select(0, positions))


def pad_masking(x, target_len):
    # x: (batch_size, seq_len)
    batch_size, seq_len = x.size()
    padded_positions = x == PAD_TOKEN_INDEX  # (batch_size, seq_len)
    pad_mask = padded_positions.unsqueeze(1).expand(batch_size, target_len, seq_len)
    return pad_mask


def subsequent_masking(x):
    # x: (batch_size, seq_len - 1)
    batch_size, seq_len = x.size()
    subsequent_mask = np.triu(np.ones(shape=(seq_len, seq_len)), k=1).astype('uint8')
    subsequent_mask = torch.tensor(subsequent_mask).to(x.device)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
    return subsequent_mask
