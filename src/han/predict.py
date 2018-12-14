# -*- coding: utf-8 -*-
"""
@author: 
        - Ardalan Mehrani <ardalan77400@gmail.com>
@brief:
"""

import os
import re
import torch
import argparse
import torch.nn.functional as F

from colr import color, Colr
from torch.utils.data import DataLoader, Dataset

from net import HAN
from lib import Preprocessing, Vectorizer, tuple_batch, Vectorizer


def get_args():
    parser = argparse.ArgumentParser("""visualize word and sentence attention weights""")
    parser.add_argument("--model_path", type=str, default="/home/ardalan.mehrani/projects/nlp-benchmarks-github/models/han/ag_news/model_epoch_30")
    parser.add_argument("--classes", nargs='+', default=["World", "Sports", "Business", "Sci/Tech"])
    parser.add_argument("--sentences", type=str, default="Google engineers created a new useless shit Hopefully<sep>Facebook researchers manage to make it useful :)")
    args = parser.parse_args()
    return args


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


class Prediction():
    def __init__(self, net, preprocessor, vectorizer):
        self.net = net
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        
    def predict_with_attention(self, sentences):
        """
        sentences: list of string
        
        returns:
        y_probs: numpy prob array
        w_atts: word attention vector
        s_atts: sent attention vector
        """
           
        self.sentences_splitted = list(self.preprocessor.transform(sentences))
        self.sequences = list(self.vectorizer.transform(self.sentences_splitted))
        labels = [0] * len(self.sequences)
        loader = iter(DataLoader(TupleLoader(self.sequences, labels), batch_size=1, shuffle=False, num_workers=0, collate_fn=tuple_batch))
        
        ty_probs, w_atts, s_atts = [], [], []
        
        
        net.eval()
        for batch_t,r_t,sent_order,ls,lr,review in loader: 
            data = (batch_t,r_t,sent_order)
            
            out = self.net(data[0],data[2],ls,lr)
            
            ty_probs.append(F.softmax(out, 1))
            
            watt = net._reorder_sent(self.net.word.attention_vector.squeeze(-1).transpose(0,1), sent_order).squeeze(0)
            satt = self.net.sent.attention_vector.squeeze(-1)
            
            w_atts.append(watt)
            s_atts.append(satt)

        ty_probs = torch.cat(ty_probs, 0)
        y_probs = ty_probs.detach().cpu().numpy()
        
        return y_probs, w_atts, s_atts

    def cprint(self, words, values, color=[0,128,0]):
        assert len(words) == len(values)
        result = []
        
        for w, prob in zip(words, values):
            w_colorized = Colr(w, fore=(255, 255, 255), back=(255*prob, 0, 0))
            result.append(str(w_colorized))
        return " ".join(result)


if __name__ == "__main__":

    opt = get_args()
    # print(vars(opt))

    model_path = opt.model_path
    classes = opt.classes
    sentences = opt.sentences.split("<sep>")

    # loading model dict
    state = torch.load(model_path)

    # get mapping word to embedding index
    wdict = state["txt_dict"]

    # load network and weights
    net = HAN(ntoken=len(state["txt_dict"]),emb_size=state["embed.weight"].size(1),hid_size=state["sent.rnn.weight_hh_l0"].size(1),num_class=state["lin_out.weight"].size(0))
    del state["txt_dict"]
    net.load_state_dict(state)

    # get prediction and attention vectors
    predictor = Prediction(net, Preprocessing(), Vectorizer(word_dict=wdict))
    y_prob, watts, satts = predictor.predict_with_attention(sentences)

    # display
    output = ""
    for sentences, watt, satt, prob in zip(predictor.sentences_splitted, watts, satts, y_prob):
        label = classes[prob.argmax()]
        output += "prediction: {}: \n".format(label)
        colored_sentence=""
        for sentence, w_att, s_att in zip(sentences, watt, satt):
            w_att = w_att[w_att>0]
            colored_sent = Colr("  ", fore=(255, 255, 255), back=(0, 0, 255*s_att))
            colored_sentence += "{} {}\n".format(colored_sent, predictor.cprint(sentence, w_att))
        output += "{}\n\n".format(colored_sentence)
    print(output)
