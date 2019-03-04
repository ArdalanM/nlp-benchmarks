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
from sklearn import utils, metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# multiprocessing workaround
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

from src.datasets import load_datasets
from src.transformer.net import Model
from src.transformer.lib import Preprocessing, Vectorizer, list_to_bytes, list_from_bytes


def get_args():
    parser = argparse.ArgumentParser("""paper: Attention Is All You Need (https://arxiv.org/abs/1706.03762)""")
    parser.add_argument("--dataset", type=str, default='ag_news')
    parser.add_argument("--data_folder", type=str, default="datasets/ag_news/transformer")
    parser.add_argument("--model_folder", type=str, default="models/transformer/ag_news")
    parser.add_argument("--embedding_dim", type=int, default=128, help="")
    parser.add_argument("--attention_dim", type=int, default=64, help="")
    parser.add_argument("--n_heads", type=int, default=4, help="")
    parser.add_argument("--n_layers", type=int, default=6, help="")
    parser.add_argument("--maxlen", type=int, default=1000, help="truncate longer sequence while training")
    parser.add_argument("--dropout", type=float, default=0.1, help="")
    parser.add_argument("--n_warmup_step", type=int, default=4000, help="")
    parser.add_argument("--batch_size", type=int, default=64, help="number of example read by the gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--snapshot_interval", type=int, default=10, help="Save model every n epoch")
    parser.add_argument('--gpuid', type=int, default=1, help="select gpu indice (default = -1 = no gpu used")
    parser.add_argument('--nthreads', type=int, default=8, help="number of cpu threads")
    args = parser.parse_args()
    return args


def get_metrics(cm, list_metrics):
    """Compute metrics from a confusion matrix (cm)
    cm: sklearn confusion matrix
    returns:
    dict: {metric_name: score}

    """
    dic_metrics = {}
    total = np.sum(cm)

    if 'accuracy' in list_metrics:
        out = np.sum(np.diag(cm))
        dic_metrics['accuracy'] = out/total

    if 'pres_0' in list_metrics:
        num = cm[0, 0]
        den = cm[:, 0].sum()
        dic_metrics['pres_0'] =  num/den if den > 0 else 0

    if 'pres_1' in list_metrics:
        num = cm[1, 1]
        den = cm[:, 1].sum()
        dic_metrics['pres_1'] = num/den if den > 0 else 0

    if 'recall_0' in list_metrics:
        num = cm[0, 0]
        den = cm[0, :].sum()
        dic_metrics['recall_0'] = num/den if den > 0 else 0

    if 'recall_1' in list_metrics:
        num = cm[1, 1]
        den = cm[1, :].sum()
        dic_metrics['recall_1'] =  num/den if den > 0 else 0

    return dic_metrics


def train(epoch,net,dataset,device,msg="val/test",optimize=False,optimizer=None,criterion=None):
    
    net.train() if optimize else net.eval()

    epoch_loss = 0
    nclasses = len(list(net.parameters())[-1])
    cm = np.zeros((nclasses,nclasses), dtype=int)

    with tqdm(total=len(dataset),desc="Epoch {} - {}".format(epoch, msg)) as pbar:
        for iteration, (tx,tidx,ty) in enumerate(dataset):

            data = (tx,tidx,ty)
            data = [x.to(device) for x in data]

            if optimize:
                optimizer.zero_grad()

            out = net(data[0],data[1])
            ty_prob = F.softmax(out, 1) # probabilites

            #metrics
            y_true = ty.detach().cpu().numpy()
            y_pred = ty_prob.max(1)[1].cpu().numpy()

            cm += metrics.confusion_matrix(y_true, y_pred, labels=range(nclasses))
            dic_metrics = get_metrics(cm, list_metrics)
            
            loss =  criterion(out, data[2]) 
            epoch_loss += loss.item()
            dic_metrics['logloss'] = epoch_loss/(iteration+1)

            if optimize:
                dic_metrics['lr'] = optimizer._optimizer.state_dict()['param_groups'][0]['lr']

            if optimize:
                loss.backward()
                optimizer.step_and_update_lr()
                
            pbar.update(1)
            pbar.set_postfix(dic_metrics)


def predict(net,dataset,device,msg="prediction"):
    
    net.eval()

    y_probs, y_trues = [], []

    for iteration, (batch_t,r_t,sent_order,ls,lr,review) in tqdm(enumerate(dataset), total=len(dataset), desc="{}".format(msg)):

        data = (batch_t,r_t,sent_order)
        data = [x.to(device) for x in data]
        out = net(data[0],data[2],ls,lr)
        ty_prob = F.softmax(out, 1) # probabilites
        y_probs.append(ty_prob.detach().cpu().numpy())
        y_trues.append(r_t.detach().cpu().numpy())

    return np.concatenate(y_probs, 0), np.concatenate(y_trues, 0).reshape(-1, 1)


def save(net, txt_dict, path):
    """
    Saves a model's state and it's embedding dic by piggybacking torch's save function
    """
    dict_m = net.state_dict()
    dict_m["txt_dict"] = txt_dict
    torch.save(dict_m,path)


def collate_fn(l):
    
    Xs, idxs, y = zip(*l)
    maxlen = max(map(len, Xs))
    Xs = [np.pad(x, (0, maxlen-len(x)), 'constant') for x in Xs]
    idxs = [np.pad(idx, (0, maxlen-len(idx)), 'constant') for idx in idxs]

    tx = torch.LongTensor(Xs)
    tidx = torch.LongTensor(idxs)
    ty = torch.LongTensor(y)
    
    return tx,tidx,ty


class TupleLoader(Dataset):

    def __init__(self, path="", maxlen=1000):
        self.path = path
        self.env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.maxlen = maxlen

    def __len__(self):
        return list_from_bytes(self.txn.get('nsamples'.encode()))[0]

    def __getitem__(self, i):

        """
        i: int
        xtxt: np.array([maxlen])
        xidx = np.array([maxlen])
        """

        xtxt = list_from_bytes(self.txn.get(('txt-%09d' % i).encode()), np.int)
        lab = list_from_bytes(self.txn.get(('lab-%09d' % i).encode()), np.int)[0]
        
        xtxt = xtxt[:self.maxlen]
                
        # getting position index of each word 
        xidx = np.array([idx+1 if w != 0 else 0 for idx, w in enumerate(xtxt)])
        
        return xtxt, xidx, lab


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr



if __name__ == "__main__":

    opt = get_args()

    os.makedirs(opt.model_folder, exist_ok=True)
    os.makedirs(opt.data_folder, exist_ok=True)

    print("parameters: {}".format(vars(opt)))

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
        tr_sentences = [txt for txt, lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
        te_sentences = [txt for txt, lab in tqdm(dataset.load_test_data(), desc="counting test samples")]
            
        n_tr_samples = len(tr_sentences)
        n_te_samples = len(te_sentences)

        print("[{}/{}] train/test samples".format(n_tr_samples, n_te_samples))
        
        prepro = Preprocessing()
        vecto = Vectorizer()
        
        ################ 
        # fit on train #
        ################
        for sentence, label in tqdm(dataset.load_train_data(), desc="fit on train...", total= n_tr_samples):    
            vecto.partial_fit(prepro.transform(sentence))

        del tr_sentences
        del te_sentences
        ###################
        # transform train #
        ###################
        with lmdb.open(variables['train']['path'], map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(tqdm(dataset.load_train_data(), desc="transform train...", total= n_tr_samples)):

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
                for i, (sentence, label) in enumerate(tqdm(dataset.load_test_data(), desc="transform test...", total= n_te_samples)):

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

    tr_loader = DataLoader(TupleLoader(variables['train']['path'], maxlen=opt.maxlen), batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=opt.nthreads, pin_memory=True)
    te_loader = DataLoader(TupleLoader(variables['test']['path'], maxlen=opt.maxlen), batch_size=opt.batch_size, collate_fn=collate_fn,  shuffle=False, num_workers=opt.nthreads, pin_memory=False)
    
    # select cpu or gpu
    device = torch.device("cuda:{}".format(opt.gpuid) if opt.gpuid >= 0 else "cpu")
    list_metrics = ['accuracy', 'pres_0', 'pres_1', 'recall_0', 'recall_1']


    print("Creating model...")
    net = Model(n_classes=n_classes, n_layers=opt.n_layers, embedding_max_index=n_tokens, max_sequence_length=longuest_sequence,
                embedding_dim=opt.embedding_dim, attention_dim=opt.attention_dim, n_heads=opt.n_heads, dropout=opt.dropout, position_wise_hidden_size=2048)

    criterion = torch.nn.CrossEntropyLoss()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    net.to(device)

    optimizer = ScheduledOptim(torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()),
                               betas=(0.9, 0.98), eps=1e-09),opt.embedding_dim, opt.n_warmup_step)


    for epoch in range(1, opt.epochs + 1):
        train(epoch,net, tr_loader, device, msg="training", optimize=True, optimizer=optimizer, criterion=criterion)
        train(epoch,net, te_loader, device, msg="testing ", criterion=criterion)

        if (epoch % opt.snapshot_interval == 0) and (epoch > 0):
            path = "{}/model_epoch_{}".format(opt.model_folder,epoch)
            print("snapshot of model saved as {}".format(path))
            save(net,variables['params']['var'], path=path)


    if opt.epochs > 0:
        path = "{}/model_epoch_{}".format(opt.model_folder,opt.epochs)
        print("snapshot of model saved as {}".format(path))
        save(net,variables['params']['var'], path=path)
