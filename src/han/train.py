# -*- coding: utf-8 -*-
"""
@author: 
        - Charles-Emmanuel DIAS  <Charles-Emmanuel.Dias@lip6.fr>
        - Ardalan Mehrani <ardalan77400@gmail.com>
@brief:
"""

import os
import re
import torch
import lmdb
import pickle
import argparse
import numpy as np
import torch.nn as nn
import pickle as pkl

from tqdm import tqdm
from sklearn import utils, metrics
from src.datasets import load_datasets

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.han.net import HAN
from src.han.lib import Preprocessing, Vectorizer, tuple_batch, Vectorizer, TupleLoader, list_to_bytes, list_from_bytes

# multiprocessing workaround
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def get_args():
    parser = argparse.ArgumentParser("""
    paper: Hierarchical Attention Network (https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
    credit to cedias: https://github.com/cedias/Hierarchical-Sentiment
    """)
    parser.add_argument("--dataset", type=str, default='ag_news')
    parser.add_argument("--data_folder", type=str, default="datasets/ag_news/han")
    parser.add_argument("--model_folder", type=str, default="models/han/ag_news")
    parser.add_argument("--solver_type", type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument("--batch_size", type=int, default=32, help="number of example read by the gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_halve_interval", type=int, default=-1, help="Number of iterations before halving learning rate")
    parser.add_argument("--gamma", type=float, default=0.1, help="learning halving facor")
    parser.add_argument("--snapshot_interval", type=int, default=2, help="Save model every n epoch")
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


def train(epoch,net,dataset,device,msg="val/test",optimize=False,optimizer=None,scheduler=None,criterion=None):
    
    net.train() if optimize else net.eval()

    epoch_loss = 0
    nclasses = len(list(net.parameters())[-1])
    cm = np.zeros((nclasses,nclasses), dtype=int)

    with tqdm(total=len(dataset),desc="Epoch {} - {}".format(epoch, msg)) as pbar:
        for iteration, (batch_t,r_t,sent_order,ls,lr,review) in enumerate(dataset):

            data = (batch_t,r_t,sent_order)
            data = [x.to(device) for x in data]

            if optimize:
                optimizer.zero_grad()

            out = net(data[0],data[2],ls,lr)
            ty_prob = F.softmax(out, 1) # probabilites

            #metrics
            y_true = r_t.detach().cpu().numpy()
            y_pred = ty_prob.max(1)[1]

            cm += metrics.confusion_matrix(y_true, y_pred, labels=range(nclasses))
            dic_metrics = get_metrics(cm, list_metrics)
            
            loss =  criterion(out, data[1]) 
            epoch_loss += loss.item()
            dic_metrics['logloss'] = epoch_loss/(iteration+1)

            if optimize:
                loss.backward()
                optimizer.step()
                dic_metrics['lr'] = optimizer.state_dict()['param_groups'][0]['lr']

            pbar.update(1)
            pbar.set_postfix(dic_metrics)

    if scheduler:
        scheduler.step()


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
        'txt_dict': {'var': None, 'path': "{}/txt_dict.pkl".format(opt.data_folder)},
    }

    # check if datasets exis
    all_exist = True if os.path.exists(variables['txt_dict']['path']) else False

    if all_exist:
        variables['txt_dict']['var'] = pkl.load(open(variables['txt_dict']['path'],"rb"))
        n_tokens = len(variables['txt_dict']['var'])

    else:
        print("Creating datasets")
        tr_sentences = [txt for txt, lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
        te_sentences = [txt for txt, lab in tqdm(dataset.load_test_data(), desc="counting test samples")]
            
        n_tr_samples = len(tr_sentences)
        n_te_samples = len(te_sentences)

        print("[{}/{}] train/test samples".format(n_tr_samples, n_te_samples))
        
        prepro = Preprocessing(batch_size=opt.batch_size, n_threads=opt.nthreads)
        vecto = Vectorizer()
        
        ################ 
        # fit on train #
        ################
        for sentence, label in tqdm(dataset.load_train_data(), desc="fit on train...", total= n_tr_samples):
            
            sentence_prepro = prepro.transform([sentence])
            vecto.partial_fit(sentence_prepro)

        del tr_sentences
        del te_sentences
        ###################
        # transform train #
        ###################
        with lmdb.open(variables['train']['path'], map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(tqdm(dataset.load_train_data(), desc="transform train...", total= n_tr_samples)):

                    xtxt = vecto.transform(prepro.transform([sentence]))[0]
                    lab = label

                    lab_key = 'lab-%09d' % i
                    txn.put(lab_key.encode(), list_to_bytes([lab]))

                    for j, l in enumerate(xtxt):
                        txt_key = 'txt-%09d-{}'.format(j) % i
                        txn.put(txt_key.encode(), list_to_bytes(l))
                    key = 'txtn-%09d' % i
                    dtxtn = txn.put(key.encode(), list_to_bytes([len(xtxt)]))
                txn.put('nsamples'.encode(), list_to_bytes([i+1]))

        ##################
        # transform test #
        ##################
        with lmdb.open(variables['test']['path'], map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(tqdm(dataset.load_test_data(), desc="transform test...", total= n_te_samples)):

                    xtxt = vecto.transform(prepro.transform([sentence]))[0]
                    lab = label

                    lab_key = 'lab-%09d' % i
                    txn.put(lab_key.encode(), list_to_bytes([lab]))

                    for j, l in enumerate(xtxt):
                        txt_key = 'txt-%09d-{}'.format(j) % i
                        txn.put(txt_key.encode(), list_to_bytes(l))
                    key = 'txtn-%09d' % i
                    dtxtn = txn.put(key.encode(), list_to_bytes([len(xtxt)]))
                txn.put('nsamples'.encode(), list_to_bytes([i+1]))

        variables['txt_dict']['var'] = vecto.word_dict
        n_tokens = len(variables['txt_dict']['var'])

        ###############
        # saving data #
        ###############     
        print("  - saving to {}".format(variables['txt_dict']['path']))
        pkl.dump(variables['txt_dict']['var'],open(variables['txt_dict']['path'],"wb"))
        
    tr_loader = DataLoader(TupleLoader(variables['train']['path'], nthreads=None), batch_size=opt.batch_size, shuffle=False, num_workers=opt.nthreads, collate_fn=tuple_batch, pin_memory=True)
    te_loader = DataLoader(TupleLoader(variables['test']['path'], nthreads=None),  batch_size=opt.batch_size, shuffle=False, num_workers=opt.nthreads, collate_fn=tuple_batch, pin_memory=False)

    # select cpu or gpu
    device = torch.device("cuda:{}".format(opt.gpuid) if opt.gpuid >= 0 else "cpu")
    list_metrics = ['accuracy', 'pres_0', 'pres_1', 'recall_0', 'recall_1']


    print("Creating model...")
    net = HAN(num_class=n_classes, ntoken=n_tokens, emb_size=200, hid_size=50)

    criterion = torch.nn.CrossEntropyLoss()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    net.to(device)

    if opt.solver_type == 'sgd':
        print(" - optimizer: sgd")
        optimizer = torch.optim.SGD(net.parameters(), lr = opt.lr)    
    elif opt.solver_type == 'adam':
        print(" - optimizer: adam")
        optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr)

    scheduler = None
    if opt.lr_halve_interval > 0:
        print(" - lr scheduler: {}".format(opt.lr_halve_interval))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_halve_interval, gamma=opt.gamma, last_epoch=-1)
        
    for epoch in range(1, opt.epochs + 1):
        train(epoch,net, tr_loader, device, msg="training", optimize=True, optimizer=optimizer, scheduler=scheduler, criterion=criterion)
        train(epoch,net, te_loader, device, msg="testing ", criterion=criterion)

        if (epoch % opt.snapshot_interval == 0) and (epoch > 0):
            path = "{}/model_epoch_{}".format(opt.model_folder,epoch)
            print("snapshot of model saved as {}".format(path))
            save(net,n_tokens, path=path)


    if opt.epochs > 0:
        path = "{}/model_epoch_{}".format(opt.model_folder,opt.epochs)
        print("snapshot of model saved as {}".format(path))
        save(net,n_tokens, path=path)
