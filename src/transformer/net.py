# -*- coding: utf-8 -*-
"""
@author: Yu-Hsiang Huang (source: https://github.com/jadore801120/attention-is-all-you-need-pytorch)
         Ardalan MEHRANI <ardalan77400@gmail.com>
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q, masking_value=0):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(masking_value) # [batch_size, len_seq_k]
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # [batch_size, len_seq_q, len_seq_k]

    return padding_mask


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        
        # q=k=v for nlp tasks 
        # q: [batch_size, seq_len_q, embedding_dim]
        # k: [batch_size, seq_len_k, embedding_dim]
        # v: [batch_size, seq_len_v, embedding_dim]

        # mask: # [batch_size, len_seq_q, len_seq_k]
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        batch_size, len_q, _ = q.size()
        batch_size, len_k, _ = k.size()
        batch_size, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k) # [batch_size, len_q, n_head, d_k]
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k) # [batch_size, len_k, n_head, d_k]
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v) # [batch_size, len_v, n_head, d_v]

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # [n_head * batch_size, len_q, d_k]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # [n_head * batch_size, len_k, d_k]
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # [n_head * batch_size, len_v, d_v]

        mask = mask.repeat(n_head, 1, 1) # [n_head * batch_size, seq_len_q, seq_len_k]
        output, attn = self.attention(q, k, v, mask=mask) # [n_head * batch_size, seq_len_q, d_v], [n_head * batch_size, seq_len_q, seq_len_q]

        output = output.view(n_head, batch_size, len_q, d_v) # [n_head, batch_size, seq_len_q, d_v]
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1) # [batch_size, seq_len_q, n_head * d_v]

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask) # [batch_size, seq_len, emb_dim]
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Model(nn.Module):

    def __init__(self,n_classes=2, n_layers=3, max_sequence_length=200, embedding_max_index=2000, embedding_dim=100,
                 attention_dim=64, n_heads=10, dropout=0.1, position_wise_hidden_size=2048):
        super().__init__()

        self.n_classes = n_classes
        self.n_layers = n_layers
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.embedding_max_index = embedding_max_index
        self.attention_dim = attention_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.position_wise_hidden_size = position_wise_hidden_size

        self.word_emb = nn.Embedding(self.embedding_max_index + 1, self.embedding_dim, padding_idx=0)

        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(self.max_sequence_length + 1, self.embedding_dim, padding_idx=0),freeze=True)


        self.layers = []
        for _ in range(n_layers):
            self.layers.append(EncoderLayer(d_model=self.embedding_dim, d_inner=self.position_wise_hidden_size,
                                            n_head=self.n_heads, d_k=self.attention_dim, d_v=self.attention_dim, dropout=self.dropout))
        self.layers = nn.ModuleList(self.layers)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classification_layer = nn.Linear(self.embedding_dim, self.n_classes)

    def forward(self, batch_sequence, batch_position):
    
        # mask padded tokens
        slf_attn_mask = get_attn_key_pad_mask(batch_sequence, batch_sequence, masking_value=0) # [batch_size, seq_len_q, seq_len_q] 
        
        # mask existing tokens
        non_pad_mask = get_non_pad_mask(batch_sequence) # [batch_size, seq_len, embedding_dim]

        # positional encoding
        enc_output = self.word_emb(batch_sequence) + self.pos_emb(batch_position) # [batch_size, seq_len, emb_dim]
            
        for layer in self.layers:
            enc_output, enc_self_attn = layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask) # [batch_size, seq_len, emb_dim]
        
        enc_output = enc_output.permute(0, 2, 1) # [batch_size, emb_dim, seq_len]
        output = self.pooling(enc_output).squeeze(-1) # [batch_size, emb_dim]
        output = self.classification_layer(output) # [batch_size, n_classes]
        return output
