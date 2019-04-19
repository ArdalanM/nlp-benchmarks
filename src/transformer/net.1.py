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


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


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
    
    def __init__(self, h, d_model, d_ff, dropout=0.1, n_layer=2):
        super(Encoders, self).__init__()
        
        self.n_layer = n_layer
        self.layers = clones(EncoderLayer(h, d_model, d_ff, dropout), self.n_layer)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."

        for layer in self.layers:
            x = layer(x, mask)
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
    def __init__(self, h, d_model, d_ff, dropout=0.1, n_layer=2):
        super(Decoders, self).__init__()
        
        self.n_layer = n_layer
        self.decoder_layers = clones(DecoderLayer(h, d_model, d_ff, dropout), self.n_layer)
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):

        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)



class EncoderDecoder(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout=0.1, n_layer=1):
        super(EncoderDecoder, self).__init__()
        self.encoder_layers = Encoders(h, d_model, d_ff, dropout, n_layer)
        self.decoder_layers = Decoders(h, d_model, d_ff, dropout, n_layer)

    def forward(self, src, tgt, src_mask, tgt_mask):
        
        enc_output = self.encoder_layers(src, src_mask)
        dec_output = self.decoder_layers(tgt, enc_output, src_mask, tgt_mask)
        return dec_output


class LMTransformer(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size, h, d_model, d_ff, dropout=0.1, n_layer=1):
        super(LMTransformer, self).__init__()
        self.pe = PositionalEncoding(d_model, dropout, max_len=5000) # arbitrary big max_len
        self.src_emb = Embeddings(d_model, src_vocab_size)
        self.trg_emb = Embeddings(d_model, trg_vocab_size)
        self.transformer = EncoderDecoder(h, d_model, d_ff, dropout=dropout, n_layer=n_layer)
        self.proj = nn.Linear(d_model, trg_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_emb = self.pe(self.src_emb(src))
        tgt_emb = self.pe(self.trg_emb(tgt))

        h = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        out = self.proj(h)
        out = F.log_softmax(out, dim=-1)
        return out


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0, device=None):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.device = device

        if device:
            self.src = self.src.to(device)
            self.src_mask = self.src_mask.to(device)

        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

            if device:
                self.trg = self.trg.to(device)
                self.trg_y = self.trg_y.to(device)
                self.trg_mask = self.trg_mask.to(device)
                self.ntokens = self.ntokens.to(device)


    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def data_gen(V, batch, nbatches, device):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0, device=device)


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

d_model=32
N=4
d_ff=2048
h=8
dropout=0.1

nclasses = 11
batch_size = 30
n_batches = 10000

gpuid = 1
device = torch.device("cuda:{}".format(gpuid) if gpuid >= 0 else "cpu")

gen = data_gen(nclasses, batch_size, n_batches, device)
# ntokens = int(batch.ntokens)

model = LMTransformer(nclasses, nclasses, h, d_model, d_ff, dropout=dropout, n_layer=N)
optimizer = NoamOpt(model.src_emb.d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
criterion = SmoothLoss(nclasses, padding_idx=0, smoothing=0)

model.to(device)
criterion.to(device)

n_iter = 100000
for i in range(n_iter):
    try:
        batch = next(gen)
    except:
        gen = data_gen(nclasses, batch_size, n_batches, device)
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
src = Variable(torch.FloatTensor([[1,2,3,4,5,6,7,8,9,10]])).to(device)
src_mask = Variable(torch.ones(1, 1, src.size(1))).to(device)
# print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

memory = model.transformer.encoder_layers(model.src_emb(src), src_mask)


ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
for i in range(max_len-1):
    out = model.decode(memory, src_mask, 
                        Variable(ys), 
                        Variable(subsequent_mask(ys.size(1))
                                .type_as(src.data)))
    prob = model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim = 1)
    next_word = next_word.data[0]
    ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)








# For data loading.
from torchtext import data, datasets
import spacy
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                    eos_token = EOS_WORD, pad_token=BLANK_WORD)

MAX_LEN = 100
train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT),  filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
MIN_FREQ = 2
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch, device):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx,device)


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)



# GPUs to use
# devices = [0, 1, 2, 3]
pad_idx = TGT.vocab.stoi["<blank>"]
BATCH_SIZE = 32
# train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)
# valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)
# model_par = nn.DataParallel(model, device_ids=devices)


model = LMTransformer(len(SRC.vocab), len(TGT.vocab), h=8, d_model=128, d_ff=1024, dropout=0.1, n_layer=3)
optimizer = NoamOpt(model.src_emb.d_model, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
criterion = SmoothLoss(nclasses=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)

model.to(device)
criterion.to(device)

model.train()

for epoch in range(10):
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=None, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)

    for i, batch in enumerate(train_iter):
        batch = rebatch(pad_idx, batch, device)
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

