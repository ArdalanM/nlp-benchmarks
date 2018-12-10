# -*- coding: utf-8 -*-
"""
@author: DIAS Charles-Emmanuel <Charles-Emmanuel.Dias@lip6.fr>
@author: MEHRANI Ardalan <ardalan77400@gmail.com>
"""

import lmdb
import torch
import spacy
import numpy as np

from collections import Counter
from torch.utils.data import DataLoader, Dataset

class Preprocessing():

    def __init__(self, batch_size=None, n_threads=8, dataset=None, language='en'):
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.language = language

        self.dataset=dataset
        
        self.nlp = spacy.load(self.language, disable=['tagger', 'parser', 'ner', 'tokenizer','tensorizer', 'similarity', 'textcat', 'sbd', 'merge_noun_chunks', 'merge_entities'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def transform(self, sentences):
        """
        sentences: list(str) iterator
        output: list(list(str)) iterator
        """
        reviews = self.nlp.pipe(sentences, batch_size=self.batch_size, n_threads=self.n_threads)
        
        out = []
        for doc in reviews:
            # out.append([[w.text for w in s] for s in doc.sents])
            out.append([[w for w in s.split()] for candidate in doc.sents for s in candidate.text.splitlines() if len(s) > 0])
        return out

 
class Vectorizer():
    def __init__(self,word_dict=None):
        self.word_dict = word_dict
        self.word_counter = Counter()

        self.n_transform = 0

        if self.word_dict:
            self.n_transform += 1
    
    def partial_fit(self, lll):
        """
        ll: list of list
        """
        for ll in lll:
            for l in ll:
                self.word_counter.update(l)

    def transform(self,lll,trim=True):
        """
        lll: list(list(list(int)))
        list of review, review is a list of sequences, sequences is a list of int
        """

        if self.n_transform == 0:
            
            # "We only retain words appearing more than 5 times in building the vocabulary and replace the words that appear 5 times with a special UNK token"
            self.word_counter = {k : v for k,v in self.word_counter.items() if v > 5}
            self.word_dict =  {w: i for i,w in enumerate(self.word_counter, start=2)}
            self.word_dict["_pad_"] = 0
            self.word_dict["_unk_"] = 1
            print("Dictionnary has {} words".format(len(self.word_dict)))
        self.n_transform += 1
        
        assert self.word_dict, "No dictionnary to vectorize text \n-> call method build_dict \n-> or set a word_dict attribute \n first"

        reviews = []
        for rev in lll:
            review = []
            for j,sent in enumerate(rev):  
                s = []
                for k,word in enumerate(sent):
                    
                    s.append(self.word_dict.get(word, self.word_dict["_unk_"]))
                
                if len(s) >= 1:
                    review.append(s)
            if len(review) == 0:
                review = [[self.word_dict["_unk_"]]]        
            reviews.append(review)
        return reviews


class TupleLoader(Dataset):

    def __init__(self, path="", nthreads=None):
        self.path = path

        self.env = lmdb.open(path, max_readers=nthreads, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return list_from_bytes(self.txn.get('nsamples'.encode()))[0]

    def __getitem__(self, i):
        nsentence = list_from_bytes(self.txn.get(('txtn-%09d' % i).encode()))[0]

        xtxt = [list_from_bytes(self.txn.get(('txt-%09d-{}'.format(j) % i).encode()), dtype=np.int).tolist() for j in range(nsentence)]
        lab = list_from_bytes(self.txn.get(('lab-%09d' % i).encode()), np.int)[0]
        return xtxt, lab


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


def list_to_bytes(l):
    return np.array(l).tobytes()


def list_from_bytes(string, dtype=np.int):
    return np.frombuffer(string, dtype=dtype)
