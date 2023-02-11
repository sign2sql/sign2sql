import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# encoder
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MyDecoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(MyDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers[:-1]:
            x, _ = layer(x, memory, src_mask, tgt_mask)
        x, last_hidden = self.layers[-1](x, memory, src_mask, tgt_mask)
        return self.norm(x), last_hidden


class MyDecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(MyDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        hidden = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](hidden, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward), self.sublayer[1].norm(hidden)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.record_first_batch = True

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


##################################################################
# new module for DSA
##################################################################
class CascadeMultiHeadedAttention(nn.Module):
    """
    query -> memory1 -> memory2
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(CascadeMultiHeadedAttention, self).__init__()
        self.multihead_attn1 = MultiHeadedAttention(h, d_model, dropout)
        self.multihead_attn2 = MultiHeadedAttention(h, d_model, dropout)

    def forward(self, query, key1, value1, key2, value2, mask1=None, mask2=None):
        query = self.multihead_attn1(query, key1, value1, mask1)
        query = self.multihead_attn2(query, key2, value2, mask2)
        return query


class CascadeDecoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(CascadeDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory1, src_mask1, memory2, src_mask2, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory1, src_mask1, memory2, src_mask2, tgt_mask)
        return self.norm(x)


class CascadeDecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(CascadeDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory1, src_mask1, memory2, src_mask2, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m1 = memory1
        m2 = memory2
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m1, m1, m2, m2, src_mask1, src_mask2))
        return self.sublayer[2](x, self.feed_forward)


##################################################################
# new module for Hybrid Cascade Multi-head Attention
##################################################################
class HybridCascadeMultiHeadedAttention(nn.Module):
    """
    query -> memory1
                     + -> memory3
    query -> memory2
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(HybridCascadeMultiHeadedAttention, self).__init__()
        self.multihead_attn1 = MultiHeadedAttention(h, d_model, dropout)
        # self.multihead_attn2 = MultiHeadedAttention(h, d_model, dropout)  # share weights or not?
        self.multihead_attn3 = MultiHeadedAttention(h, d_model, dropout)

    def forward(self, query, key1, value1, key2, value2, key3, value3, mask1=None, mask2=None, mask3=None):
        attn1 = self.multihead_attn1(query, key1, value1, mask1)
        attn2 = self.multihead_attn1(query, key2, value2, mask2)  # share weights or not?
        query = attn1 + attn2
        query = self.multihead_attn3(query, key3, value3, mask3)
        return query


class HybridCascadeDecoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(HybridCascadeDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory1, src_mask1, memory2, src_mask2, memory3, src_mask3, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory1, src_mask1, memory2, src_mask2, memory3, src_mask3, tgt_mask)
        return self.norm(x)


class HybridCascadeDecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(HybridCascadeDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory1, src_mask1, memory2, src_mask2, memory3, src_mask3, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m1 = memory1
        m2 = memory2
        m3 = memory3
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m1, m1, m2, m2, m3, m3, src_mask1, src_mask2, src_mask3))
        return self.sublayer[2](x, self.feed_forward)


##################################################################
# new module for Link Attention (Learnable Keys)
##################################################################
class LinkDecoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(LinkDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, key, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, key, memory, src_mask, tgt_mask)
        return self.norm(x)


class LinkDecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(LinkDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, key, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, key, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


##################################################################


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class ConvReluConvFFN(nn.Module):
    def __init__(self, d_model, d_ff, first_kernel_size=9, second_kernel_size=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, (first_kernel_size,), padding=((first_kernel_size - 1) // 2,))
        self.conv2 = nn.Conv1d(d_ff, d_model, (second_kernel_size,), padding=((second_kernel_size - 1) // 2,))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.conv2(self.dropout(F.relu(self.conv1(x.permute((0, 2, 1)))))).permute((0, 2, 1))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

    def forward_prob(self, x_prob):
        # x_prob: [B, T, vocab]
        return torch.matmul(x_prob, self.lut.weight) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def make_transformer_encoder(N_layer=4, d_model=256, d_ff=1024, heads=8, dropout=0.1, ffn_layer='conv_relu_conv',
                             first_kernel_size=9):
    "Helper: Construct a transformer encoder from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(heads, d_model)
    if ffn_layer == 'conv_relu_conv':
        ff = ConvReluConvFFN(d_model, d_ff, first_kernel_size, second_kernel_size=1, dropout=dropout)
    else:
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_layer)
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)
    return model


def make_transformer_decoder(N_layer=4, d_model=256, d_ff=1024, heads=8, dropout=0.1, ffn_layer='conv_relu_conv',
                             first_kernel_size=9):
    "Helper: Construct a transformer decoder from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(heads, d_model)
    if ffn_layer == 'conv_relu_conv':
        ff = ConvReluConvFFN(d_model, d_ff, first_kernel_size, second_kernel_size=1, dropout=dropout)
    else:
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N_layer)
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)
    return model


def make_my_transformer_decoder(N_layer=4, d_model=256, d_ff=1024, heads=8, dropout=0.1, ffn_layer='conv_relu_conv',
                             first_kernel_size=9):
    "Helper: Construct a transformer decoder from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(heads, d_model)
    if ffn_layer == 'conv_relu_conv':
        ff = ConvReluConvFFN(d_model, d_ff, first_kernel_size, second_kernel_size=1, dropout=dropout)
    else:
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = MyDecoder(MyDecoderLayer(d_model, c(attn), c(attn),
                                     c(ff), dropout), N_layer)
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)
    return model


def make_cascade_transformer_decoder(N_layer=4, d_model=256, d_ff=1024, heads=8, dropout=0.1, ffn_layer='conv_relu_conv',
                             first_kernel_size=9):
    c = copy.deepcopy
    self_attn = MultiHeadedAttention(heads, d_model)
    cascade_attn = CascadeMultiHeadedAttention(heads, d_model)
    if ffn_layer == 'conv_relu_conv':
        ff = ConvReluConvFFN(d_model, d_ff, first_kernel_size, second_kernel_size=1, dropout=dropout)
    else:
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = CascadeDecoder(CascadeDecoderLayer(d_model, c(self_attn), c(cascade_attn),
                                               c(ff), dropout), N_layer)
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)
    return model


def make_hybrid_cascade_transformer_decoder(N_layer=4, d_model=256, d_ff=1024, heads=8, dropout=0.1, ffn_layer='conv_relu_conv',
                                            first_kernel_size=9):
    c = copy.deepcopy
    self_attn = MultiHeadedAttention(heads, d_model)
    hybrid_cascade_attn = HybridCascadeMultiHeadedAttention(heads, d_model)
    if ffn_layer == 'conv_relu_conv':
        ff = ConvReluConvFFN(d_model, d_ff, first_kernel_size, second_kernel_size=1, dropout=dropout)
    else:
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = HybridCascadeDecoder(HybridCascadeDecoderLayer(d_model, c(self_attn), c(hybrid_cascade_attn),
                                                           c(ff), dropout), N_layer)
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)
    return model


def make_link_transformer_decoder(N_layer=4, d_model=256, d_ff=1024, heads=8, dropout=0.1, ffn_layer='conv_relu_conv',
                             first_kernel_size=9):
    "Helper: Construct a transformer decoder from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(heads, d_model)
    if ffn_layer == 'conv_relu_conv':
        ff = ConvReluConvFFN(d_model, d_ff, first_kernel_size, second_kernel_size=1, dropout=dropout)
    else:
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = LinkDecoder(LinkDecoderLayer(d_model, c(attn), c(attn),
                                         c(ff), dropout), N_layer)
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform(p)
            nn.init.xavier_uniform_(p)
    return model


if __name__ == '__main__':
    encoder = make_transformer_encoder()
    x = torch.randn(24, 75, 256)
    # x_mask = torch.randint(0, 2, (24, 75))==1
    hidden = encoder.forward(x, None)
    print(hidden.shape)
    # decoder = make_transformer_decoder()
    # y = decoder.forward(x, hidden, x_mask.unsqueeze(-2), x_mask.unsqueeze(-2))
    # print(y.shape)
    # sz=5
    # print(subsequent_mask(sz))
    # mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    # print(mask)
    # x_mask = torch.randint(0, 2, (3, 5))==1
    # print(x_mask)
    # mask = x_mask.unsqueeze(-2) & subsequent_mask(x_mask.size(-1))
    # print(mask)
