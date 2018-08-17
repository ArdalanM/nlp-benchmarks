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
import spacy
import itertools
import argparse
import numpy as np
import torch.nn as nn
import pickle as pkl

from tqdm import tqdm
from collections import Counter
from sklearn import utils, metrics
from gensim.models import KeyedVectors
from src import lib
from src.datasets import load_datasets


from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from src.han.net import HAN


def get_args():
    parser = argparse.ArgumentParser("""
    paper: Hierarchical Attention Network (https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
    credit to cedias: https://github.com/cedias/Hierarchical-Sentiment
    """)
    parser.add_argument("--dataset", type=str, default='imdb')
    parser.add_argument("--data_folder", type=str, default="datasets/imdb/han")
    parser.add_argument("--model_folder", type=str, default="models/han/imdb")
    parser.add_argument("--pretrain",action="store_true", default=True, help="pre train embeddings with word2vec (gensim required)")
    parser.add_argument('--max_feats', type=int, default=10000, help="vocabulary size")
    parser.add_argument('--max_sents', type=int, default=-1, help="max number of sentences per example")
    parser.add_argument('--max_words', type=int, default=-1, help="max word length")
    parser.add_argument("--solver_type", type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument("--batch_size", type=int, default=32, help="number of example read by the gpu")
    parser.add_argument("--test_batch_size", type=int, default=None, help="batch size during prediction (equals to batch_size if none)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_halve_interval", type=int, default=-1, help="Number of iterations before halving learning rate")
    parser.add_argument("--gamma", type=float, default=0.1, help="learning halving facor")
    parser.add_argument("--snapshot_interval", type=int, default=10, help="Save model every n epoch")
    parser.add_argument("--model_weights_path", type=str, default="")
    parser.add_argument('--gpuid', type=int, default=0, help="select gpu indice (default = -1 = no gpu used")
    parser.add_argument('--nthreads', type=int, default=4, help="number of cpu threads")
    args = parser.parse_args()
    return args


class Preprocessing():

    def __init__(self, batch_size=None, n_threads=8, delimiters=(';', '\n', '.')):
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.delimiters = [re.escape(x) for x in delimiters]
        self.pattern = re.compile("|".join(self.delimiters))


    def transform(self, sentences):
        """
        sentences: list(str) iterator
        output: list(list(str)) iterator
        """
        
        for review in sentences:
            yield [sentence.split() for sentence in re.split(self.pattern, review) if len(sentence) > 0]


class Vectorizer():
    def __init__(self,word_dict=None, max_sent_len=8, max_word_len=32):
        self.word_dict = word_dict
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len
    
    def fit(self, text_iterator, max_words):
        word_counter = Counter(itertools.chain.from_iterable(w for s in tqdm(text_iterator, desc="counting words") for w in s))
        self.word_dict =  {w: i for i,(w,_) in tqdm(enumerate(word_counter.most_common(max_words),start=2),desc="building word dict",total=max_words)}
        self.word_dict["_pad_"] = 0
        self.word_dict["_unk_"] = 1
        print("Dictionnary has {} words".format(len(self.word_dict)))
    
    def transform(self,t,trim=True):

        if self.word_dict is None:
            print("No dictionnary to vectorize text \n-> call method build_dict \n-> or set a word_dict attribute \n first")
            raise Exception

        if type(t) == str:
            t = [t]

        if self.max_sent_len < 0 and self.max_word_len < 0:
            trim = False

        for rev in t:
            review = []
            for j,sent in enumerate(rev):  

                if trim and j>= self.max_sent_len:
                    break
                s = []
                for k,word in enumerate(sent):

                    if trim and k >= self.max_word_len:
                        break

                    if word in self.word_dict:
                        s.append(self.word_dict[word])
                    else:
                        s.append(self.word_dict["_unk_"]) #_unk_word_
                
                if len(s) >= 1:
                    review.append(s)
            if len(review) == 0:
                review = [[self.word_dict["_unk_"]]]        
            # revs.append(review)
            yield review
        # return revs


def tuple_batch(l):
    """
    Prepare batch
    - Reorder reviews by length
    - Split reviews by sentences which are reordered by length
    - Build sentence ordering index to extract each sentences in training loop
    """
    review, rating = zip(*l)
    r_t = torch.Tensor(rating).long()
    list_rev = review

    sorted_r = sorted([(len(r),r_n,r) for r_n,r in enumerate(list_rev)],reverse=True) #index by desc rev_le
    lr,r_n,ordered_list_rev = zip(*sorted_r)
    lr = list(lr)
    max_sents = lr[0]

    #reordered
    r_t = r_t[[r_n]]
    review = [review[x] for x in r_n]

    stat =  sorted([(len(s),r_n,s_n,s) for r_n,r in enumerate(ordered_list_rev) for s_n,s in enumerate(r)],reverse=True)
    max_words = stat[0][0]

    ls = []
    batch_t = torch.zeros(len(stat),max_words).long()                         # (sents ordered by len)
    sent_order = torch.zeros(len(ordered_list_rev),max_sents).long().fill_(0) # (rev_n,sent_n)

    for i,s in enumerate(stat):
        sent_order[s[1],s[2]] = i+1 #i+1 because 0 is for empty.
        batch_t[i,0:len(s[3])] = torch.LongTensor(s[3])
        ls.append(s[0])

    return batch_t, r_t, sent_order, ls, lr, review


class TupleLoader(Dataset):

    def __init__(self, sequences, labels):
        assert len(sequences) == len(labels)
        
        self.sequences = sequences
        self.labels = labels
        self.len = len(self.labels)

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return (self.sequences[i], self.labels[i])


def yield_index(loader, i=0):
    for _ in loader:
        yield _[i]


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
            
            if optimize:
                loss =  criterion(out, data[1]) 
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                dic_metrics['logloss'] = epoch_loss/(iteration+1)
                dic_metrics['lr'] = optimizer.state_dict()['param_groups'][0]['lr']

            pbar.update(1)
            pbar.set_postfix(dic_metrics)

    if scheduler:
        scheduler.step()
    logger.info(dic_metrics)


def save(net,dic,path):
    """
    Saves a model's state and it's embedding dic by piggybacking torch's save function
    """
    dict_m = net.state_dict()
    dict_m["word_dic"] = dic    
    torch.save(dict_m,path)


def load_embeddings(filepath, offset=0):
    wv = KeyedVectors.load_word2vec_format(filepath, binary=False)

    wdict = {word:i for i, word in tqdm(enumerate(wv.index2word, offset), desc="word indexing", total=len(wv.index2word))}
    tensor = torch.FloatTensor(wv.vectors)
    tensor = F.pad(tensor, (0,0,offset,0)) # add 2 row of zeros for __unk__ and __pad__ tokens
    return tensor, wdict


if __name__ == "__main__":

    opt = get_args()
    opt.test_batch_size =  opt.test_batch_size if opt.test_batch_size else opt.batch_size

    os.makedirs(opt.model_folder, exist_ok=True)
    os.makedirs(opt.data_folder, exist_ok=True)

    logger = lib.get_logger(logdir=opt.model_folder, logname="logs.txt")
    logger.info("parameters: {}".format(vars(opt)))

    dataset = load_datasets(names=[opt.dataset])[0]
    dataset_name = dataset.__class__.__name__
    n_classes = dataset.n_classes
    logger.info("dataset: {}, n_classes: {}".format(dataset_name, n_classes))


    tr_seq_path = "{}/train_sequences.pkl".format(opt.data_folder)
    te_seq_path = "{}/test_sequences.pkl".format(opt.data_folder)

    tr_lab_path = "{}/train_labels.pkl".format(opt.data_folder)
    te_lab_path = "{}/test_labels.pkl".format(opt.data_folder)

    wdict_path = "{}/word_dict.pkl".format(opt.data_folder)

    embedding_path = "{}/word_embeddings".format(opt.data_folder)


    # check if datasets exist
    all_exist = True
    l = [tr_seq_path, te_seq_path, tr_lab_path, te_lab_path, wdict_path]
    if opt.pretrain:
        l.append(embedding_path)
    for path in l:
        if not os.path.exists(path):
            all_exist = False

    if all_exist:

        logger.info("Loading existing dataset: ")
        logger.info("  - Loading: {}".format(tr_seq_path))
        tr_seq = pkl.load(open(tr_seq_path,"rb"))

        logger.info("  - Loading: {}".format(tr_lab_path))
        tr_lab = pkl.load(open(tr_lab_path,"rb"))
        
        logger.info("  - Loading: {}".format(te_seq_path))
        te_seq = pkl.load(open(te_seq_path,"rb"))

        logger.info("  - Loading: {}".format(te_lab_path))
        te_lab = pkl.load(open(te_lab_path,"rb"))

        if opt.pretrain:
            logger.info("  - loading {}".format(embedding_path))
            tensor, wdict = load_embeddings(embedding_path, offset=2)
            wdict["_pad_"] = 0
            wdict["_unk_"] = 1
            n_tokens = len(wdict)
        
        else:
            logger.info("  - Loading: {}".format(wdict_path))
            wdict = pkl.load(open(wdict_path,"rb"))
            n_tokens = len(wdict)


    else:
        logger.info("Creating datasets")
        logger.info("  - loading raw datasets")
        tr_data = dataset.load_train_data()
        te_data = dataset.load_test_data()
        
        prepro = Preprocessing(batch_size=opt.batch_size)
        vecto = Vectorizer(max_sent_len=opt.max_sents, max_word_len=opt.max_words)
        
        if opt.pretrain:
            logger.info("  -unsupervised pre training")
            import gensim

            tr_rev_iter = itertools.chain.from_iterable(yield_index(dataset.load_train_data(), 0))
            te_rev_iter = itertools.chain.from_iterable(yield_index(dataset.load_test_data(), 0))
            rev_iter = itertools.chain(tr_rev_iter, te_rev_iter) # sentence iterator
            logger.info("  - loading sentences in to ram :(")
            sentences = [sent.split() for sent in rev_iter]

            logger.info("  - train word2vec")
            w2vmodel = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=5, iter=2, max_vocab_size=10000000, workers=opt.nthreads)
            
            logger.info("  - save embbedings: {}".format(embedding_path))
            w2vmodel.wv.save_word2vec_format(embedding_path,total_vec=len(w2vmodel.wv.vocab))  
            
            logger.info("  - loading embedding from {}".format(embedding_path))
            tensor, wdict = load_embeddings(embedding_path, offset=2)
            wdict["_pad_"] = 0
            wdict["_unk_"] = 1
            n_tokens = len(wdict)
            vecto.word_dict = wdict
        
        else:
        
            logger.info("  - fit")
            tr_sentences = itertools.chain.from_iterable(yield_index(dataset.load_train_data(), 0))
            vecto.fit(prepro.transform(tr_sentences), max_words=opt.max_feats)
            wdict = vecto.word_dict
            n_tokens = len(wdict)

        logger.info("  - transform train")
        tr_sentences = itertools.chain.from_iterable(yield_index(dataset.load_train_data(), 0))
        tr_seq = list(vecto.transform(prepro.transform(tr_sentences)))
        tr_lab = list(itertools.chain.from_iterable(yield_index(dataset.load_train_data(), 1)))


        logger.info("  - transform test")
        te_sentences = itertools.chain.from_iterable(yield_index(dataset.load_test_data(), 0))
        te_seq = list(vecto.transform(prepro.transform(te_sentences)))
        te_lab = list(itertools.chain.from_iterable(yield_index(dataset.load_test_data(), 1)))

        logger.info("  - saving datasets")
        
        logger.info("  - saving to {}".format(tr_seq_path))
        pkl.dump(tr_seq,open(tr_seq_path,"wb"))
        
        logger.info("  - saving to {}".format(te_seq_path))
        pkl.dump(te_seq,open(te_seq_path,"wb"))

        logger.info("  - saving to {}".format(tr_lab_path))
        pkl.dump(tr_lab,open(tr_lab_path,"wb"))

        logger.info("  - saving to {}".format(te_lab_path))
        pkl.dump(te_lab,open(te_lab_path,"wb"))

        logger.info("  - saving to {}".format(wdict_path))
        pkl.dump(wdict,open(wdict_path,"wb"))


    tr_loader = DataLoader(TupleLoader(tr_seq, tr_lab), batch_size=opt.batch_size, shuffle=True, num_workers=opt.nthreads, collate_fn=tuple_batch, pin_memory=True)
    te_loader = DataLoader(TupleLoader(te_seq, te_lab), batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.nthreads, collate_fn=tuple_batch)

    # select cpu or gpu
    device = torch.device("cuda:{}".format(opt.gpuid) if opt.gpuid >= 0 else "cpu")
    list_metrics = ['accuracy', 'pres_0', 'pres_1', 'recall_0', 'recall_1']


    logger.info("Creating model...")
    if opt.model_weights_path and os.path.exists(opt.model_weights_path):
        logger.info(" --loading existing weights from: {}".format(opt.model_weights_path))
        state = torch.load(opt.model_weights_path)
        wdict = state["word_dic"]
        n_tokens = len(wdict)

        net = HAN(ntoken=len(state["word_dic"]),emb_size=state["embed.weight"].size(1),hid_size=state["sent.rnn.weight_hh_l0"].size(1),num_class=state["lin_out.weight"].size(0))
        del state["word_dic"]
        net.load_state_dict(state)

    else:
        net = HAN(n_tokens, n_classes, emb_size=200, hid_size=50)

    if opt.pretrain:
        net.set_emb_tensor(tensor)

    criterion = torch.nn.CrossEntropyLoss()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    net.to(device)

    if opt.solver_type == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr = opt.lr)    
    elif opt.solver_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr)

    scheduler = None
    if opt.lr_halve_interval > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_halve_interval, gamma=opt.gamma, last_epoch=-1)
        

    for epoch in range(1, opt.epochs + 1):
        train(epoch,net, tr_loader, device, msg="training", optimize=True, optimizer=optimizer, scheduler=scheduler, criterion=criterion)
        train(epoch,net, te_loader, device, msg="testing ")

        if (epoch % opt.snapshot_interval == 0) and (epoch > 0):
            path = "{}/model_epoch_{}".format(opt.model_folder,epoch)
            print("snapshot of model saved as {}".format(path))
            save(net, wdict, path=path)


    path = "{}/model_epoch_{}".format(opt.model_folder,opt.epochs)
    print("snapshot of model saved as {}".format(path))
    save(net, wdict, path=path)
