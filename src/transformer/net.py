# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model)
        div_term = torch.exp(div_term)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],  requires_grad=False)
        return self.dropout(x)


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


class EncoderLayer(nn.Module):
    
    "Encoder is made up of self-attn and feed forward (defined below)"
    
    def __init__(self, h, d_model, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.d = dropout
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.n1 = LayerNorm(d_model)
        self.n2 = LayerNorm(d_model)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."

        attn_output = self.self_attn(self.n1(x), self.n1(x), self.n1(x), mask)
        attn_output = F.dropout(attn_output, self.d)
        attn_output = x + attn_output

        enc_output = self.feed_forward(self.n2(attn_output))
        enc_output = F.dropout(enc_output, self.d)
        enc_output = attn_output + enc_output

        return enc_output


class Encoders(nn.Module):
    
    "Encoder is made up of self-attn and feed forward (defined below)"
    
    def __init__(self, vocab_size, h, d_model, d_ff, dropout=0.1, n_layer=2):
        super(Encoders, self).__init__()
        self.pe = PositionalEncoding(d_model, dropout, max_len=5000) # arbitrary big max_len
        self.emb = Embeddings(d_model, vocab_size)
        self.layers = clones(EncoderLayer(h, d_model, d_ff, dropout), n_layer)
        self.norm = LayerNorm(d_model)

    def forward(self, src, src_mask):
        "Follow Figure 1 (left) for connections."

        x = self.pe(self.emb(src))

        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.d = dropout
        self.dec_attn = MultiHeadedAttention(h, d_model, dropout)
        self.enc_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.n1 = LayerNorm(d_model) 
        self.n2 = LayerNorm(d_model) 
        self.n3 = LayerNorm(d_model) 
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        

        # attention over target words + residual connection
        attn_output = self.dec_attn(self.n1(x), self.n1(x), self.n1(x), tgt_mask)
        attn_output = F.dropout(attn_output, self.d)
        attn_output = attn_output + x

        # attention between target words and source words + residual connection
        attn_input = self.enc_attn(self.n2(attn_output), self.n2(enc_output), self.n2(enc_output), src_mask)
        attn_input = F.dropout(attn_input, self.d)
        attn_input = attn_input + attn_output

        dec_output = self.feed_forward(self.n3(attn_input))
        dec_output = F.dropout(dec_output, self.d)
        dec_output = dec_output + attn_input    

        return dec_output


class Decoders(nn.Module):
    def __init__(self, vocab_size, h, d_model, d_ff, dropout=0.1, n_layer=2):
        super(Decoders, self).__init__()

        self.pe = PositionalEncoding(d_model, dropout, max_len=5000) # arbitrary big max_len
        self.emb = Embeddings(d_model, vocab_size)
        self.decoder_layers = clones(DecoderLayer(h, d_model, d_ff, dropout), n_layer)
        self.norm = LayerNorm(d_model)
    
    def forward(self, src, enc_output, src_mask, tgt_mask):
        
        x = self.pe(self.emb(src))

        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


class SequenceClsHead(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(SequenceClsHead, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.proj(x)


class ClsHead(nn.Module):

    def __init__(self, input_dim, num_labels):
        super(ClsHead, self).__init__()
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj1 = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        #     # We "pool" the model by simply taking the hidden state corresponding
        #     # to the first token. We assume that this has been pre-trained
        out = x[:, 0]

        out = torch.tanh(self.proj(out))
        out = self.proj1(out)
        return out


class TransformerLM(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size, h, d_model, d_ff, dropout=0.1, n_layer=1):
        super(TransformerLM, self).__init__()
        self.encoders = Encoders(src_vocab_size, h, d_model, d_ff, dropout=dropout, n_layer=n_layer)
        self.decoders = Decoders(trg_vocab_size, h, d_model, d_ff, dropout=dropout, n_layer=n_layer)
        self.classification = SequenceClsHead(d_model, trg_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):

        enc_output = self.encoders(src, src_mask)
        dec_output = self.decoders(tgt, enc_output, src_mask, tgt_mask)
        out = self.classification(dec_output)
        return out


class TransformerCls(nn.Module):

    def __init__(self, nclasses, src_vocab_size, h, d_model, d_ff, dropout=0.1, n_layer=1):
        super(TransformerCls, self).__init__()
        self.encoders = Encoders(src_vocab_size, h, d_model, d_ff, dropout=dropout, n_layer=n_layer)
        self.classification = ClsHead(d_model, nclasses)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask):

        enc_output = self.encoders(src, src_mask)
        out = self.classification(enc_output)
        return out


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0, device=None):

        self.src = src.to(device)
        self.src_mask = src.ne(pad).unsqueeze(-2).to(device)
        self.device = device

        if trg is not None:

            self.trg = trg.to(device)
            if trg.dim() > 1:
                self.trg = trg[:, :-1].to(device)
                self.trg_y = trg[:, 1:].to(device)
                self.trg_mask = self.make_std_mask(self.trg, pad)
                self.ntokens = (self.trg_y != pad).data.sum().to(device)

    @staticmethod  
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = tgt.ne(pad).unsqueeze(1)
        next_word_mask = subsequent_mask(tgt.size(-1)).to(tgt_mask.device)
        tgt_mask = tgt_mask & next_word_mask
        return tgt_mask


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    return subsequent_mask.eq(0)


def data_gen_binary_classification(batch_size, nbatches, device, vocab_size):
    "Generate random data for a src-tgt copy task."
    seq_length = 100
    thresh = (seq_length * vocab_size) // 2
    for i in range(nbatches):
        tx = torch.from_numpy(np.random.randint(1, vocab_size, size=(batch_size, seq_length)))
        ty = (tx.sum(-1) > thresh).type_as(tx.data)
        yield Batch(tx, ty, 0, device=device)


def data_gen_sequence(vocab_size, batch, nbatches, device):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, vocab_size, size=(batch, 10))
        data[:, 0] = 1
        yield Batch(data, data, 0, device=device)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,  Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        

class SmoothLoss(nn.Module):
    "Implement label smoothing."
    def __init__(self, nclasses=None, padding_idx=0, smoothing=0.0):
        super(SmoothLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.nclasses = nclasses
        
    def forward(self, ty_prob, ty_true):
        assert ty_true.size(0) == ty_prob.size(0)

        if self.smoothing > 0:

            nclasses = self.nclasses or ty_prob.size(1)

            ty_true_oh = torch.zeros_like(ty_prob).fill_(self.smoothing / (nclasses - 2))
            ty_true_oh.scatter_(1, ty_true.data.unsqueeze(1), self.confidence)
            mask = torch.nonzero(ty_true == self.padding_idx)
            if mask.numel() > 0:
                ty_true_oh.index_fill_(0, mask.squeeze(), 0.0)
            loss = self.criterion(ty_prob, Variable(ty_true_oh, requires_grad=False))
        else:
            loss = F.cross_entropy(ty_prob, ty_true, ignore_index=self.padding_idx, reduction='sum')

        return loss


def test_train_tranformer_lm():

    from types import SimpleNamespace
    opt = SimpleNamespace(d_model=32, N=2, d_ff=1024, h=8, dropout=0.1, vocab_size=11, batch_size=128, n_batches=1000, gpuid=1)
    device = torch.device("cuda:{}".format(opt.gpuid) if opt.gpuid >= 0 else "cpu")
    
    gen = data_gen_sequence(opt.vocab_size, opt.vocab_size, opt.n_batches, device)
    # ntokens = int(batch.ntokens)

    model = TransformerLM(opt.vocab_size, opt.vocab_size, opt.h, opt.d_model, opt.d_ff, dropout=opt.dropout, n_layer=opt.N)
    optimizer = NoamOpt(model.encoders.emb.d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    criterion = SmoothLoss(nclasses, padding_idx=0, smoothing=0)

    model.to(device)
    criterion.to(device)

    n_iter = 1000
    for i in range(n_iter):
        try:
            batch = next(gen)
        except:
            gen = data_gen(nclasses, batch_size, opt.n_batches, device)
            batch = next(gen)

        ty_prob = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        ty_prob_reshaped = ty_prob.contiguous().view(-1, ty_prob.size(-1))
        
        ty_true = batch.trg_y
        ty_true_reshaped = ty_true.contiguous().view(-1)

        loss = criterion(ty_prob_reshaped, ty_true_reshaped)
        loss.backward()
        optimizer.step()
        optimizer.optimizer.zero_grad()
        acc = (ty_prob_reshaped.argmax(-1) == ty_true_reshaped).float().mean()

        if i > 0 and i % 50 == 1:
            print(i, acc.cpu().numpy(), loss.detach().cpu().numpy() / batch.ntokens.cpu().numpy())

    # prediction
    model.eval()
    max_len=10

    src = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]).to(device)
    src_mask = torch.ones(1, 1, src.size(1)).to(device)
    # print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

    memory = model.encoders(src, src_mask)
    ys = torch.ones(1, 1).type_as(src.data).to(device)
    trg_mask = subsequent_mask(ys.size(1)).type_as(src.data).to(device)

    for i in range(max_len-1):
        dec_output = model.decoders(ys, memory, src_mask, trg_mask)
        prob = model.classification(dec_output[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    
    print("input: {}".format(src))
    print("prediction: {}".format(ys))


def test_train_tranformer_cls():

    from types import SimpleNamespace
    opt = SimpleNamespace(nclasses=2, d_model=32, N=2, d_ff=1024, h=8, dropout=0.1, vocab_size=1000, batch_size=128, n_batches=1000, gpuid=1)
    device = torch.device("cuda:{}".format(gpuid) if gpuid >= 0 else "cpu")

    model = TransformerCls(nclasses, vocab_size, h, d_model, d_ff, dropout=dropout, n_layer=N)
    # optimizer = NoamOpt(model.encoders.emb.d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    criterion = CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    n_iter = 1000
    gen = data_gen_binary_classification(batch_size, n_batches, device, vocab_size)
    for i in range(n_iter):
        try:
            batch = next(gen)
        except:
            gen = data_gen_binary_classification(batch_size, n_batches, device, vocab_size)
            batch = next(gen)

        out = model(batch.src, batch.src_mask)
        ty_true = batch.trg

        loss = criterion(out, ty_true)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = (out.argmax(-1) == ty_true).sum().item() / ty_true.size(0)

        if i > 0 and i % 50 == 1:
            print("iter: {}, acc: {}, loss: {}".format(i, acc, loss.item()))


if __name__ == "__main__":
    test_train_tranformer_lm()
    test_train_tranformer_cls


