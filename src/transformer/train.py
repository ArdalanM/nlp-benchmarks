# -*- coding: utf-8 -*-
"""
@author: Ardalan Mehrani <ardalan77400@gmail.com>

@brief:
"""

import os
import lmdb
import argparse
import numpy as np
import pickle as pkl

from tqdm import tqdm
from sklearn import metrics
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# multiprocessing workaround
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

from src.datasets import load_datasets
from src.transformer.net import TransformerCls
from src.transformer.lib import Preprocessing, Vectorizer, list_to_bytes, list_from_bytes


def get_args():
    parser = argparse.ArgumentParser("""paper: Attention Is All You Need (https://arxiv.org/abs/1706.03762)""")
    parser.add_argument("--dataset", type=str, default='ag_news')
    parser.add_argument("--data_folder", type=str, default="datasets/ag_news/transformer")
    parser.add_argument("--model_folder", type=str, default="models/transformer/ag_news")
    
    # preprocessing
    parser.add_argument("--word_min_count", type=int, default=5, help="")
    parser.add_argument('--curriculum', default=False, action='store_true', help="curriculum learning, sort training set by lenght")

    #model
    parser.add_argument("--attention_dim", type=int, default=16, help="")
    parser.add_argument("--n_heads", type=int, default=2, help="")
    parser.add_argument("--n_layers", type=int, default=2, help="")
    parser.add_argument("--maxlen", type=int, default=20, help="truncate longer sequence while training")
    parser.add_argument("--dropout", type=float, default=0.1, help="")
    parser.add_argument("--ff_hidden_size", type=int, default=128, help="point wise feed forward nn")

    #optimizer
    parser.add_argument("--opt_name", type=str, default='adam_warmup_linear', choices=['adam', 'adam_warmup_linear'])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--n_warmup_step", type=int, default=1000, help="scheduling optimizer warmup step. set to -1 for regular adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=None, help="gradient clipping")

    # training    
    parser.add_argument("--batch_size", type=int, default=32, help="number of example read by the gpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--snapshot_interval", type=int, default=10, help="Save model every n epoch")
    parser.add_argument('--gpuid', type=int, default=0, help="select gpu index. -1 to select cpu")
    parser.add_argument('--nthreads', type=int, default=8, help="number of cpu threads")
    parser.add_argument('--use-all-gpu', default=False, action='store_true')
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    return args


def train(epoch,net,dataset,device,msg="val/test",optimize=False,optimizer=None,criterion=None):
    
    net.train() if optimize else net.eval()

    epoch_loss = 0
    epoch_acc = 0
    dic_metrics= {'loss':0, 'acc':0, 'lr':0}
    nclasses = len(list(net.parameters())[-1])

    with tqdm(total=len(dataset),desc="Epoch {} - {}".format(epoch, msg)) as pbar:
        for iteration, (tx,mask,ty) in enumerate(dataset):

            data = (tx,mask,ty)
            data = [x.to(device) for x in data]

            if optimize:
                optimizer.zero_grad()

            out = net(data[0],data[1])
            loss =  criterion(out, data[2])

            #metrics
            epoch_loss += loss.item()
            epoch_acc += (data[-1] == out.argmax(-1)).sum().item() / len(out)

            dic_metrics['loss'] = epoch_loss/(iteration+1)
            dic_metrics['acc'] = epoch_acc/(iteration+1)
            
            if optimize:
                loss.backward()
                dic_metrics['lr'] = optimizer.state_dict()['param_groups'][0]['lr']
                optimizer.step()
                
            pbar.update(1)
            pbar.set_postfix(dic_metrics)


def save(net, txt_dict, path):
    """
    Saves a model's state and it's embedding dic by piggybacking torch's save function
    """
    dict_m = net.state_dict()
    dict_m["txt_dict"] = txt_dict
    torch.save(dict_m,path)


def collate_fn(l):
    
    sequence, labels = zip(*l)
    local_maxlen = max(map(len, sequence))

    Xs = [np.pad(x, (0, local_maxlen-len(x)), 'constant') for x in sequence]
    tx = torch.LongTensor(Xs)
    tx_mask = tx.ne(0).unsqueeze(-2)
    ty = torch.LongTensor(labels) 
    return tx, tx_mask, ty


class TupleLoader(Dataset):

    def __init__(self, path=""):
        self.path = path
        self.env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return list_from_bytes(self.txn.get('nsamples'.encode()))[0]

    def __getitem__(self, i):
        """
        i: int
        xtxt: np.array([maxlen])
        """
        xtxt = list_from_bytes(self.txn.get(('txt-%09d' % i).encode()), np.int)
        lab = list_from_bytes(self.txn.get(('lab-%09d' % i).encode()), np.int)[0]
        xtxt = xtxt[:opt.maxlen]
        return xtxt, lab


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()

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


if __name__ == "__main__":

    opt = get_args()
    
    os.makedirs(opt.model_folder, exist_ok=True)
    os.makedirs(opt.data_folder, exist_ok=True)

    print("parameters:")
    pprint(vars(opt))
    torch.manual_seed(opt.seed)

    dataset = load_datasets(names=[opt.dataset])[0]
    dataset_name = dataset.__class__.__name__
    n_classes = dataset.n_classes
    print("dataset: {}, n_classes: {}".format(dataset_name, n_classes))


    variables = {
        'train': {'var': None, 'path': "{}/train.lmdb".format(opt.data_folder)},
        'test': {'var': None, 'path': "{}/test.lmdb".format(opt.data_folder)},
        'params': {'var': None, 'path': "{}/params.pkl".format(opt.data_folder)},
    }

    # check if datasets exis
    all_exist = True if os.path.exists(variables['params']['path']) else False

    if all_exist:
        variables['params']['var'] = pkl.load(open(variables['params']['path'],"rb"))
        longuest_sequence = variables['params']['var']['longest_sequence']
        n_tokens = len(variables['params']['var']['word_dict'])

    else:
        print("Creating datasets")
        tr_examples = [(txt,lab) for txt, lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
        te_examples = [(txt,lab) for txt, lab in tqdm(dataset.load_test_data(), desc="counting test samples")]
        
        if opt.curriculum:
            print(" - curriculum: sorting by sentence length")
            tr_examples = sorted(tr_examples, key=lambda r: len(r[0]), reverse=False)
            te_examples = sorted(te_examples, key=lambda r: len(r[0]), reverse=False)

        n_tr_samples = len(tr_examples)
        n_te_samples = len(te_examples)
        print(" - shortest sequence: {}, longest sequence: {}".format(len(tr_examples[0][0].split()), len(tr_examples[-1][0].split())))
        print(" - [{}/{}] train/test samples".format(n_tr_samples, n_te_samples))
        
        prepro = Preprocessing(lowercase=True)
        vecto = Vectorizer(min_word_count= opt.word_min_count)
        
        ################ 
        # fit on train #
        ################
        for sentence, label in tqdm(tr_examples, desc="fit on train...", total=n_tr_samples):    
            vecto.partial_fit(prepro.transform(sentence))

        ###################
        # transform train #
        ###################
        with lmdb.open(variables['train']['path'], map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(tqdm(tr_examples, desc="transform train...", total= n_tr_samples)):

                    xtxt = vecto.transform(prepro.transform(sentence))
                    lab = label

                    txt_key = 'txt-%09d' % i
                    lab_key = 'lab-%09d' % i
                    
                    txn.put(lab_key.encode(), list_to_bytes([lab]))
                    txn.put(txt_key.encode(), list_to_bytes(xtxt))

                txn.put('nsamples'.encode(), list_to_bytes([i+1]))

        ##################
        # transform test #
        ##################
        with lmdb.open(variables['test']['path'], map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(tqdm(te_examples, desc="transform test...", total= n_te_samples)):

                    xtxt = vecto.transform(prepro.transform(sentence))
                    lab = label

                    txt_key = 'txt-%09d' % i
                    lab_key = 'lab-%09d' % i
                    
                    txn.put(lab_key.encode(), list_to_bytes([lab]))
                    txn.put(txt_key.encode(), list_to_bytes(xtxt))

                txn.put('nsamples'.encode(), list_to_bytes([i+1]))

        variables['params']['var'] = vars(vecto)
        longuest_sequence = variables['params']['var']['longest_sequence']
        n_tokens = len(variables['params']['var']['word_dict'])

        ###############
        # saving data #
        ###############     
        print("  - saving to {}".format(variables['params']['path']))
        pkl.dump(variables['params']['var'],open(variables['params']['path'],"wb"))

    tr_loader = DataLoader(TupleLoader(variables['train']['path']), batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=opt.nthreads, pin_memory=True)
    te_loader = DataLoader(TupleLoader(variables['test']['path']), batch_size=opt.batch_size, collate_fn=collate_fn,  shuffle=False, num_workers=opt.nthreads, pin_memory=False)
    
    # select cpu or gpu
    device = torch.device("cuda:{}".format(opt.gpuid) if opt.gpuid >= 0 else "cpu")

    print("Creating model...")
    net = TransformerCls(nclasses=n_classes,
                         src_vocab_size=n_tokens,
                         h=opt.n_heads,
                         d_model=opt.attention_dim,
                         d_ff=opt.ff_hidden_size,
                         dropout=opt.dropout,
                         n_layer=opt.n_layers)
    net.to(device)

    if opt.use_all_gpu:
        print(" - Using all gpus")
        net = nn.DataParallel(net)
    
    if opt.max_grad_norm:
        print(" - gradient clipping: {}".format(opt.max_grad_norm))
        torch.nn.utils.clip_grad_norm_(net.parameters(), opt.max_grad_norm)
    
    scheduler = None
    if opt.opt_name == 'adam_warmup_linear':
        optimizer_ = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), betas=(0.9, 0.98), eps=1e-09, weight_decay=opt.weight_decay)
        optimizer = NoamOpt(opt.attention_dim, 1, opt.n_warmup_step, optimizer_)
    elif opt.opt_name == 'adam':
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_size, gamma=opt.gamma, last_epoch=-1)
    else:
        raise
    print(opt.opt_name,optimizer, scheduler)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, opt.epochs + 1):
        train(epoch,net, tr_loader, device, msg="training", optimize=True, optimizer=optimizer, criterion=criterion)
        train(epoch,net, te_loader, device, msg="testing ", criterion=criterion)
        
        if scheduler:
            scheduler.step()

        if (epoch % opt.snapshot_interval == 0) and (epoch > 0):
            path = "{}/model_epoch_{}".format(opt.model_folder,epoch)
            print("snapshot of model saved as {}".format(path))
            save(net,variables['params']['var'], path=path)


    if opt.epochs > 0:
        path = "{}/model_epoch_{}".format(opt.model_folder,opt.epochs)
        print("snapshot of model saved as {}".format(path))
        save(net,variables['params']['var'], path=path)
