# -*- coding: utf-8 -*-
"""
@author: MEHRANI Ardalan <ardalan77400@gmail.com>
"""

import os
import numpy as np


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


class CharVectorizer():
    def __init__(self, maxlen=10, alphabet="""abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]\{\}"""):
        
        self.alphabet = alphabet
        self.maxlen = maxlen
        self.char_dict = {k: i for i, k in enumerate(self.alphabet, 1)} # indice zero is reserved to blank and unknown characters

    def transform(self,sentences):
        """
        sentences: list of string
        list of review, review is a list of sequences, sequences is a list of int
        """
        sequences = []
        for sentence in sentences:
            seq = [self.char_dict.get(char, 0) for char in sentence[:self.maxlen].lower()]
            sequences.append(seq)                
        return sequences
    
    def get_params(self):
        params = vars(self)
        return params

