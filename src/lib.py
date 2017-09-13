# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
import h5py
import datetime
import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import metrics


def timestamp():
    now = datetime.datetime.now()
    now_str = now.strftime("%m%d-%H%M-%S")
    return now_str


def get_logger(logdir, logname, loglevel=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(os.path.join(logdir, logname))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def gen_from_hdf5(filename, batch_size=128, loop_infinite=True):

    f = h5py.File(filename, 'r', libver='latest')
    n_chunks = len(f.keys()) // 2

    if loop_infinite:
        while True:
            for i in range(n_chunks):
                x = f["data_{}".format(i)]
                y = f["label_{}".format(i)]
                for j in range(0, len(x), batch_size):
                    yield x[j: j + batch_size], y[j: j+batch_size]
    else:
        for i in range(n_chunks):
            x = f["data_{}".format(i)]
            y = f["label_{}".format(i)]
            for j in range(0, len(x), batch_size):
                yield x[j: j + batch_size], y[j: j+batch_size]
    f.close()


def pad_sequence(sequences, maxlen=10, padding='pre', truncating='pre', value=0):
    """
    :param sequences:
    :param maxlen:
    :param padding:
    :param truncating:
    :param value:
    :return:

     >>> sequences =[[1,3,4], [2], [1,0,3,23,65]]
     >>> pad_sequence(sequences, maxlen=3, padding='pre', truncating='pre')
     >>> [[1, 3, 4], [0, 0, 2], [3, 23, 65]]

     >>> sequences =[[1,3,4], [2], [1,0,3,23,65]]
     >>> pad_sequence(sequences, maxlen=3, padding='post', truncating='post')
     >>> [[1, 3, 4], [2, 0, 0], [1, 0, 3]]

    """



    padded_sequences = []

    for seq in sequences:

        length = len(seq)

        if truncating == 'pre':
            seq = seq[-maxlen:]
        elif truncating == 'post':
            seq = seq[:maxlen]

        if length < maxlen:

            diff = np.abs(length - maxlen)

            if padding == 'pre':
               seq = [value] * diff + seq

            elif padding == 'post':
                seq = seq + [value] * diff
        padded_sequences.append(seq)
    return padded_sequences


def create_dataset(generator, lowercase=True):

    sentences, labels = [], []

    if lowercase:
        for phrase, label in generator:
            phrase = [r.lower() for r in phrase]
            sentences.extend(phrase)
            labels.extend(label)
    else:
        for phrase, label in generator:
            sentences.extend(phrase)
            labels.extend(label)
    return sentences, labels


def get_metrics(y_true, y_prob, n_classes=2, list_metrics=[]):
    """Compute metrics for given true and predicted labels

    Parameters
    ----------
    y_true : numpy.array
        Ground true labels of items
    y_prob : numpy.array
        Predicted by model labels
    n_classes : int
        Number of labels/classes
    list_metrics: list of str
        List of metrics to compute

    Returns
    -------
    dict
        key : metric name, value: result

    """
    y_pred = np.argmax(y_prob, -1)
    dic = {}
    if 'accuracy' in list_metrics:
        dic['accuracy'] = metrics.accuracy_score(y_true, y_pred)

    if 'log_loss' in list_metrics:
        try:
            dic['logloss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            dic['logloss'] = -1

    if 'auc' in list_metrics:
        dic['auc'] = metrics.roc_auc_score(y_true, y_prob[:, 1])

    if 'pres_0' in list_metrics:
        dic['pres_0'] = metrics.precision_score(y_true, y_pred, pos_label=0)

    if 'pres_1' in list_metrics:
        dic['pres_1'] = metrics.precision_score(y_true, y_pred, pos_label=1)

    if 'recall_0' in list_metrics:
        dic['recall_0'] = metrics.recall_score(y_true, y_pred, pos_label=0)

    if 'recall_1' in list_metrics:
        dic['recall_1'] = metrics.recall_score(y_true, y_pred, pos_label=1)
    if 'cm' in list_metrics:
        dic['cm'] = str(metrics.confusion_matrix(y_true, y_pred))
    return dic


class CharOHEncoder(object):
    """
    Turn list of string into a (R x L x D) dimensional tensor:
        - R: number of row (length of the input list)
        - L: number of column (maxlen of sub lists)
        - D: Dimension of the vocabulary (length of the alphabet)
    """

    def __init__(self, alphabet=None, maxlen=10, dtype=np.int8):

        self.alphabet = alphabet
        self.maxlen = maxlen
        self.dtype = dtype

    def _encode(self, sentence, maxlen, token_indice=None):
        """
        Turn a string sentence into 2d array
        :param sentence: string sentence
        :param maxlen:
        :param token_indice:
        :return: 2d array (maxlen x len(token_indice))
        """

        if not token_indice:
            token_indice = self.token_indice

        x = np.zeros((maxlen, len(token_indice)), dtype=self.dtype)
        for t in range(min(maxlen, len(sentence))):
            token = sentence[t]
            if token in token_indice:
                x[t, token_indice[token]] = True
        return x

    def _decode(self, x, indice_token=None):
        """
        Turns a 2d array into a string sentence
        :param x:
        :param indice_token:
        :return:
        """

        if not indice_token:
            indice_token = self.indice_token

        output = []

        for array in x:
            if np.sum(array) != 0:
                index = array.argmax()
                output.append(indice_token[index])

        return "".join(output)

    def _createMappingfromAlphabet(self, alphabet):
        """"
        Mapping alphabet to dictionary
        """
        # be use tokens in alphabet are unique
        assert len(alphabet) == len(set(alphabet))

        token_indice = {v: k for k, v in enumerate(alphabet)}
        indice_token = {token_indice[k]: k for k in token_indice}

        return token_indice, indice_token

    def _assert_mapping(self, token_indice, indice_token):

        assert len(token_indice) == len(indice_token)

        for k1, k2 in zip(token_indice, indice_token):
            assert indice_token[token_indice[k1]] == k1
            assert token_indice[indice_token[k2]] == k2
        return True

    def _assert_transform(self, sentence, x, token_indice, indice_token):

        """
        Compare a sentence with the decoded version of the sentence
        :param sentence: string sentence
        :param x: 2d array of encoded sentence
        :param token_indice: dictionary {k:v} == {token:integer}
        :param indice_token: dictionary {k:v} == {integer:token}
        :return:  True/False whether both string or the same or not

        """

        sentence_label = "".join([token for token in sentence[:self.maxlen] if token in token_indice])
        sentence_candidate = self._decode(x, indice_token)
        # print(sentence_label)
        # print(sentence_candidate)
        assert sentence_label == sentence_candidate

        return True

    def transform(self, sentences, **transform_params):

        nb_features = len(self.token_indice)

        X = np.zeros((len(sentences), self.maxlen, nb_features), dtype=self.dtype)
        for i, sentence in enumerate(sentences):
            X[i] = self._encode(sentence, self.maxlen, self.token_indice)

        #sanity check
        for i in range(min(3, len(sentences))):
            assert self._assert_transform(sentences[i], X[i], self.token_indice,
                                          self.indice_token)
        return X

    def fit(self, sentences, y=None, **fit_params):

            if not self.maxlen:
                # finding longest sentence
                self.maxlen = max(list(map(len, sentences)))

            if not self.alphabet:
                #alphabet is made up of all characters
                self.alphabet = "".join(set(char for sentence in sentences for char in sentence))

            self.token_indice, self.indice_token = self._createMappingfromAlphabet(self.alphabet)
            assert self._assert_mapping(self.token_indice, self.indice_token)

            return self


class StringToSequence(object):
    """
    Turn a list of strings into a list of integers

    Example word uni-gram:
    >>> train_sentences = ["hello I am Ardalan","I like ML"]
    >>> test_sentences = ["I like unseen words", "non sense dsfasf"]
    >>> vec = StringToSequence(level="word", ngram_range=(1, 1))
    >>> vec.fit_transform(train_sentences)
    [[1, 2, 3, 4], [2, 5, 6]]
    >>> vec.transform(test_sentences)
    [[2, 5], []]
    >>> " ".join([vec.indice_token[k] for k in [1, 2, 3, 4]])
    'hello I am Ardalan'

    Example char uni-gram:
    >>> train_sentences = ["hello I am Ardalan","I like ML"]
    >>> test_sentences = ["I like unseen words", "non sense dsfasf"]
    >>> vec = StringToSequence(level="char", ngram_range=(1, 1))
    >>> vec.fit_transform(train_sentences)
    [[1, 2, 3, 3, 4, 5, 6, 5, 7, 8, 5, 9, 10, 11, 7, 3, 7, 12],
     [6, 5, 3, 13, 14, 2, 5, 15, 16]]
    >>> vec.transform(test_sentences)
    [[6, 5, 3, 13, 14, 2, 5, 12, 2, 2, 12, 5, 4, 10, 11], [12, 4, 12, 5, 2, 12, 2, 5, 11, 7]]
    """
    def __init__(self, level="word", ngram_range=(1, 1), **kvargs):
        self.level = level
        self.ngram_range = ngram_range
        self.kvargs = kvargs

        assert self.level in ["word", "char"]

    def _word_token_indice(self, sentences, ngram_range=(1, 1)):
        """
        Take sentences and return dictionary mapping ngrams >> interger (unique)

        :param ngram_range:
        :return:
        """
        min_ngrams, max_ngrams = ngram_range
        token_indice = {}
        # We make sure the minimum value of our mapping is 1 (0 will be reserved)
        indexer = 1

        for sentence in sentences:
            sentence = sentence.split(" ")
            for i in range(0, len(sentence)-max_ngrams+1, min_ngrams):
                for ngram_value in range(min_ngrams, max_ngrams+1):
                    ngram = " ".join(sentence[i:i+ngram_value])
                    if ngram not in token_indice:
                        token_indice[ngram] = indexer
                        indexer += 1

        return token_indice

    def _char_token_indice(self, sentences, ngram_range=(1, 1)):

        min_ngrams, max_ngrams = ngram_range
        token_indice = {}
        # We make sure the minimum value of our mapping is 1 (0 will be reserved)
        indexer = 1

        for sentence in sentences:
            for i in range(0, len(sentence)-max_ngrams+1, min_ngrams):
                for ngram_value in range(min_ngrams, max_ngrams+1):
                    ngram = sentence[i:i+ngram_value]
                    if ngram not in token_indice:
                        token_indice[ngram] = indexer
                        indexer += 1
        return token_indice

    def _tranform_words(self, sentences, token_indice, ngram_range=(1, 1)):

        min_ngrams, max_ngrams = ngram_range

        new_sentences = [] * len(sentences)
        for sentence in sentences:
            sentence = sentence.split(" ")
            new_sentence = []
            for i in range(0, len(sentence)-max_ngrams+1, min_ngrams):
                for ngram_value in range(min_ngrams, max_ngrams+1):
                    ngram = " ".join(sentence[i:i+ngram_value])
                    if ngram in token_indice:
                        new_sentence.append(token_indice[ngram])
            new_sentences.append(new_sentence)
        return new_sentences

    def _tranform_chars(self, sentences, token_indice, ngram_range=(1, 1)):

        min_ngrams, max_ngrams = ngram_range

        new_sentences = [] * len(sentences)
        for sentence in sentences:
            new_sentence = []
            for i in range(0, len(sentence)-max_ngrams+1, min_ngrams):
                for ngram_value in range(min_ngrams, max_ngrams+1):
                    ngram = sentence[i:i+ngram_value]
                    if ngram in token_indice:
                        new_sentence.append(token_indice[ngram])
            new_sentences.append(new_sentence)
        return new_sentences

    def inverse_tranform(self, sequences):

        new_sequences = []*len(sequences)

        for sequence in sequences:
            new_sequence = []
            for integer in sequence:
                if integer in self.indice_token:
                    new_sequence.append(self.indice_token[integer])

            if self.level == "word":
                new_sequence = " ".join(new_sequence)
            else:
                new_sequence = "".join(new_sequence)

            new_sequences.append(new_sequence)

        return new_sequences

        # if self.level == "word":



            # " ".join([vec.indice_token[val] for val in test_seq[1] if val in vec.indice_token])

    def transform(self, sentences, **transform_params):

        if self.level == "word":
            new_sentences = token_indice = self._tranform_words(sentences,
                                                     self.token_indice,
                                                     self.ngram_range)
        elif self.level == "char":
            new_sentences = self._tranform_chars(sentences,
                                                     self.token_indice,
                                                     self.ngram_range)
        else:
            raise

        return new_sentences

    def fit(self, sentences, y=None):

        if self.level == "word":
            self.token_indice = self._word_token_indice(sentences, self.ngram_range)
        elif self.level == "char":
            self.token_indice = self._char_token_indice(sentences, self.ngram_range)
        else:
            raise

        self.indice_token = {self.token_indice[k]: k for k in self.token_indice}

        return self

    def fit_transform(self, sentences):
        self.fit(sentences)
        return self.transform(sentences)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select columns of a DataFrame

    Parameters
    ----------
    column: name of a DataFrame column (string or int)
    Returns
    -------
    numpy.ndarray of that column
    """

    def __init__(self, column=None, dtype=None, return_pdseries=False, reshape=False, join_token=" "):
        """Select columns of a DataFrame

        Parameters
        ----------
        column : str or list of str
            Name of columns or list of columns to select
        dtype : numpy.type
            To which type convert column
        return_pdseries : bool
            Return data as pandas.Series, if False returns numpy.array
        reshape : bool
            If true returns numpy.ndarray
        Returns
        -------
        numpy.array or pandas.Series
            numpy.ndarray or pandas.Series of required columns
        """
        self.column = column
        self.dtype = dtype
        self.return_pdseries = return_pdseries
        self.reshape = reshape
        self.join_token = join_token

    def transform(self, df, **transform_params):
        x = df[self.column].copy()
        if self.return_pdseries:
            return x
        else:
            x = x.values
            if self.dtype:
                x = x.astype(self.dtype)
            if self.reshape:
                x = x.reshape(-1, 1)
        return x

    def fit(self, df, y=None, **fit_params):
        return self
