# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
import sys
import csv
import tarfile
import shutil
import hashlib
import pandas as pd
from urllib.request import urlretrieve
from urllib.error import URLError
from urllib.error import HTTPError
csv.field_size_limit(sys.maxsize)
DATA_FOLDER = "datasets"


def _progress(count, block_size, total_size):
    rate = float(count * block_size) / float(total_size) * 100.0
    sys.stdout.write("\r>> Downloading {:.1f}%".format(rate))


def get_file(fname, origin, untar=False, md5_hash=None, cache_subdir='datasets', check_certificate=True):
    """Downloads a file from a URL if it not already in the cache.

    Passing the MD5 hash will verify the file after download
    as well as if it is already present in the cache.

    # Arguments
        fname: name of the file
        origin: original URL of the file
        untar: boolean, whether the file should be decompressed
        md5_hash: MD5 hash of the file for verification
        cache_subdir: directory being used as the cache

    # Returns
        Path to the downloaded file
    """
    datadir_base = DATA_FOLDER
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if md5_hash is not None:
            if not validate_file(fpath, md5_hash):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)
        error_msg = 'URL fetch failure on {}: {} -- {}'

        if not check_certificate:
            import requests
            r = requests.get(origin, verify=False)
            with open(fpath, 'wb') as fd:
                fd.write(r.content)
        else:
            try:
                try:
                    urlretrieve(origin, fpath, _progress)
                    sys.stdout.flush()
                except URLError as e:
                    raise Exception(error_msg.format(origin, e.errno, e.reason))
                except HTTPError as e:
                    raise Exception(error_msg.format(origin, e.code, e.msg))
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(fpath):
                    os.remove(fpath)
                raise

    if untar:
        if not os.path.exists(untar_fpath):
            print('Untaring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise
            tfile.close()
        return untar_fpath

    return fpath


def validate_file(fpath, md5_hash):
    """Validates a file against a MD5 hash.

    # Arguments
        fpath: path to the file being validated
        md5_hash: the MD5 hash being validated against

    # Returns
        Whether the file is valid
    """
    hasher = hashlib.md5()
    with open(fpath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    if str(hasher.hexdigest()) == str(md5_hash):
        return True
    else:
        return False


class AgNews(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):

        folder = "agnews"
        csv_folder = "ag_news_csv"
        self.n_classes = 4
        self.folder_path = "{}/{}/{}".format(DATA_FOLDER, folder, csv_folder)
        self.url = "https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/ag_news_csv.tar.gz"
        self.epoch_size = 5000

        # Check if relevant files are in the folder_path
        if os.path.exists(self.folder_path):
            for f in ["classes.txt", "readme.txt", "test.csv", "train.csv"]:
                if not os.path.exists(os.path.join(self.folder_path, f)):
                    self._ = get_file(folder, origin=self.url, untar=True, cache_subdir=folder)
        else:
            self._ = get_file(folder, origin=self.url, untar=True, cache_subdir=folder)

    def load_train_test_csv(self):
        df_tr = pd.read_csv(os.path.join(self.folder_path, "train.csv"), encoding="utf-8")
        df_te = pd.read_csv(os.path.join(self.folder_path, "test.csv"), encoding="utf-8")
        return df_tr, df_te

    @staticmethod
    def _generator(filename, chunk_size=512):

        f = open(filename, mode='r', encoding='utf-8')
        reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
        while True:
            sentences, labels = [], []
            i = 0

            for line in reader:
                sentence = "{} {}".format(line['title'], line['description'])
                label = int(line['label']) - 1
                sentences.append(sentence)
                labels.append(label)
                i += 1
                if i == chunk_size:
                    i = 0
                    yield sentences, labels
                    sentences, labels = [], []

            if sentences and labels:
                yield sentences, labels
            else:
                break
        f.close()

    def load_train_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "train.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "train.csv"))

    def load_test_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "test.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "test.csv"))


class DbPedia(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):

        folder = "dbpedia"
        csv_folder = "dbpedia_csv"
        self.n_classes = 14
        self.folder_path = "{}/{}/{}".format(DATA_FOLDER, folder, csv_folder)
        self.url = "https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/dbpedia_csv.tar.gz"
        self.epoch_size = 5000

        # Check if relevant files are in the folder_path
        if os.path.exists(self.folder_path):
            for f in ["classes.txt", "readme.txt", "test.csv", "train.csv"]:
                assert os.path.exists(os.path.join(self.folder_path, f))
        else:
            self._ = get_file(folder, origin=self.url, untar=True, cache_subdir=folder)

    @staticmethod
    def _generator(filename, chunk_size=512):

        f = open(filename, mode='r', encoding='utf-8')
        reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
        while True:
            sentences, labels = [], []
            i = 0

            for line in reader:
                sentence = "{} {}".format(line['title'], line['description'])
                label = int(line['label']) - 1
                sentences.append(sentence)
                labels.append(label)
                i += 1
                if i == chunk_size:
                    i = 0
                    yield sentences, labels
                    sentences, labels = [], []

            if sentences and labels:
                yield sentences, labels
            else:
                break
        f.close()

    def load_train_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "train.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "train.csv"))

    def load_test_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "test.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "test.csv"))


class YelpReview(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):

        folder = "yelp_review"
        csv_folder = "yelp_review_full_csv"
        self.n_classes = 5
        self.folder_path = "{}/{}/{}".format(DATA_FOLDER, folder, csv_folder)
        self.url = "https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/yelp_review_full_csv.tar.gz"
        self.epoch_size = 5000

        # Check if relevant files are in the folder_path
        if os.path.exists(self.folder_path):
            for f in ["readme.txt", "test.csv", "train.csv"]:
                assert os.path.exists(os.path.join(self.folder_path, f))
        else:
            self._ = get_file(folder, origin=self.url, untar=True, cache_subdir=folder)

    @staticmethod
    def _generator(filename, chunk_size=512):

        f = open(filename, mode='r', encoding='utf-8')
        reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
        while True:
            sentences, labels = [], []
            i = 0

            for line in reader:
                sentence = "{} {}".format(line['title'], line['description'])
                label = int(line['label']) - 1
                sentences.append(sentence)
                labels.append(label)
                i += 1
                if i == chunk_size:
                    i = 0
                    yield sentences, labels
                    sentences, labels = [], []

            if sentences and labels:
                yield sentences, labels
            else:
                break
        f.close()

    def load_train_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "train.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "train.csv"))

    def load_test_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "test.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "test.csv"))
2

class YelpPolarity(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):

        folder = "yelp_review_polarity"
        csv_folder = "yelp_review_polarity_csv"
        self.n_classes = 2
        self.folder_path = "{}/{}/{}".format(DATA_FOLDER, folder, csv_folder)
        self.url = "https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/yelp_review_polarity_csv.tar.gz"
        self.epoch_size = 5000

        # Check if relevant files are in the folder_path
        if os.path.exists(self.folder_path):
            for f in ["readme.txt", "test.csv", "train.csv"]:
                assert os.path.exists(os.path.join(self.folder_path, f))
        else:
            self._ = get_file(folder, origin=self.url, untar=True, cache_subdir=folder)

    @staticmethod
    def _generator(filename, chunk_size=512):

        f = open(filename, mode='r', encoding='utf-8')
        reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
        while True:
            sentences, labels = [], []
            i = 0

            for line in reader:
                sentence = "{} {}".format(line['title'], line['description'])
                label = int(line['label']) - 1
                sentences.append(sentence)
                labels.append(label)
                i += 1
                if i == chunk_size:
                    i = 0
                    yield sentences, labels
                    sentences, labels = [], []

            if sentences and labels:
                yield sentences, labels
            else:
                break
        f.close()

    def load_train_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "train.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "train.csv"))

    def load_test_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "test.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "test.csv"))


class AmazonReview(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):

        folder = "amazon_review"
        csv_folder = "amazon_review_full_csv"
        self.n_classes = 5
        self.folder_path = "{}/{}/{}".format(DATA_FOLDER, folder, csv_folder)
        self.url = "https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/amazon_review_full_csv.tar.gz"
        self.epoch_size = 30000

        # Check if relevant files are in the folder_path
        if os.path.exists(self.folder_path):
            for f in ["readme.txt", "test.csv", "train.csv"]:
                assert os.path.exists(os.path.join(self.folder_path, f))
        else:
            self._ = get_file(folder, origin=self.url, untar=True, cache_subdir=folder)

    @staticmethod
    def _generator(filename, chunk_size=512):

        f = open(filename, mode='r', encoding='utf-8')
        reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
        while True:
            sentences, labels = [], []
            i = 0

            for line in reader:
                sentence = "{} {}".format(line['title'], line['description'])
                label = int(line['label']) - 1
                sentences.append(sentence)
                labels.append(label)
                i += 1
                if i == chunk_size:
                    i = 0
                    yield sentences, labels
                    sentences, labels = [], []

            if sentences and labels:
                yield sentences, labels
            else:
                break
        f.close()

    def load_train_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "train.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "train.csv"))

    def load_test_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "test.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "test.csv"))


class AmazonPolarity(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):

        folder = "amazon_polarity"
        csv_folder = "amazon_review_polarity_csv"
        self.n_classes = 2
        self.folder_path = "{}/{}/{}".format(DATA_FOLDER, folder, csv_folder)
        self.url = "https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/amazon_review_polarity_csv.tar.gz"
        self.epoch_size = 30000

        # Check if relevant files are in the folder_path
        if os.path.exists(self.folder_path):
            for f in ["readme.txt", "test.csv", "train.csv"]:
                assert os.path.exists(os.path.join(self.folder_path, f))
        else:
            self._ = get_file(folder, origin=self.url, untar=True, cache_subdir=folder)

    @staticmethod
    def _generator(filename, chunk_size=512):

        f = open(filename, mode='r', encoding='utf-8')
        reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
        while True:
            sentences, labels = [], []
            i = 0

            for line in reader:
                sentence = "{} {}".format(line['title'], line['description'])
                label = int(line['label']) - 1
                sentences.append(sentence)
                labels.append(label)
                i += 1
                if i == chunk_size:
                    i = 0
                    yield sentences, labels
                    sentences, labels = [], []

            if sentences and labels:
                yield sentences, labels
            else:
                break
        f.close()

    def load_train_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "train.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "train.csv"))

    def load_test_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "test.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "test.csv"))


class SoguNews(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):

        folder = "sogou_news"
        csv_folder = "sogou_news_csv"
        self.n_classes = 5
        self.folder_path = "{}/{}/{}".format(DATA_FOLDER, folder, csv_folder)
        self.url = "https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/sogou_news_csv.tar.gz"
        self.epoch_size = 5000

        # Check if relevant files are in the folder_path
        if os.path.exists(self.folder_path):
            for f in ["readme.txt", "test.csv", "train.csv"]:
                assert os.path.exists(os.path.join(self.folder_path, f))
        else:
            self._ = get_file(folder, origin=self.url, untar=True, cache_subdir=folder)

    @staticmethod
    def _generator(filename, chunk_size=512):

        f = open(filename, mode='r', encoding='utf-8')
        reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
        while True:
            sentences, labels = [], []
            i = 0

            for line in reader:
                sentence = "{} {}".format(line['title'], line['description'])
                label = int(line['label']) - 1
                sentences.append(sentence)
                labels.append(label)
                i += 1
                if i == chunk_size:
                    i = 0
                    yield sentences, labels
                    sentences, labels = [], []

            if sentences and labels:
                yield sentences, labels
            else:
                break
        f.close()

    def load_train_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "train.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "train.csv"))

    def load_test_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "test.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "test.csv"))


class YahooAnswer(object):
    """
    credit goes to Xiang Zhang:
    https://scholar.google.com/citations?hl=en&user=n4QjVfoAAAAJ&view_op=list_works&sortby=pubdate
    """
    def __init__(self):

        folder = "yahoo_answers"
        csv_folder = "yahoo_answers_csv"
        self.n_classes = 10
        self.folder_path = "{}/{}/{}".format(DATA_FOLDER, folder, csv_folder)
        self.url = "https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/yahoo_answers_csv.tar.gz"
        self.epoch_size = 10000

        # Check if relevant files are in the folder_path
        if os.path.exists(self.folder_path):
            for f in ["readme.txt", "test.csv", "train.csv"]:
                assert os.path.exists(os.path.join(self.folder_path, f))
        else:
            self._ = get_file(folder, origin=self.url, untar=True, cache_subdir=folder)

    @staticmethod
    def _generator(filename, chunk_size=512):

        f = open(filename, mode='r', encoding='utf-8')
        reader = csv.DictReader(f, fieldnames=['label', 'title', 'description'], quotechar='"')
        while True:
            sentences, labels = [], []
            i = 0

            for line in reader:
                sentence = "{} {}".format(line['title'], line['description'])
                label = int(line['label']) - 1
                sentences.append(sentence)
                labels.append(label)
                i += 1
                if i == chunk_size:
                    i = 0
                    yield sentences, labels
                    sentences, labels = [], []

            if sentences and labels:
                yield sentences, labels
            else:
                break
        f.close()

    def load_train_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "train.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "train.csv"))

    def load_test_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "test.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "test.csv"))


class Imdb(object):
    """
    source: http://ai.stanford.edu/~amaas/data/sentiment/
    """
    def __init__(self):

        folder = "imdb"
        csv_folder = "imdb_csv"
        self.n_classes = 2
        self.folder_path = "{}/{}/{}".format(DATA_FOLDER, folder, csv_folder)
        self.url = "https://s3.eu-west-2.amazonaws.com/ardalan.mehrani.datasets/imdb_csv.tar.gz"

        # Check if relevant files are in the folder_path
        if os.path.exists(self.folder_path):
            for f in ["readme.txt", "test.csv", "train.csv"]:
                assert os.path.exists(os.path.join(self.folder_path, f))
        else:
            self._ = get_file(folder, origin=self.url, untar=True, cache_subdir=folder)

    @staticmethod
    def _generator(filename, chunk_size=512):

        f = open(filename, mode='r', encoding='utf-8')
        reader = csv.DictReader(f, quotechar='"')
        while True:
            sentences, labels = [], []
            i = 0

            for line in reader:
                sentence = line['sentence']
                label = int(line['label'])
                sentences.append(sentence)
                labels.append(label)
                i += 1
                if i == chunk_size:
                    i = 0
                    yield sentences, labels
                    sentences, labels = [], []

            if sentences and labels:
                yield sentences, labels
            else:
                break
        f.close()

    def load_train_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "train.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "train.csv"))

    def load_test_data(self, chunk_size=512):
        if chunk_size:
            return self._generator(os.path.join(self.folder_path, "test.csv"), chunk_size=chunk_size)
        else:
            self._generator(os.path.join(self.folder_path, "test.csv"))


def load_datasets(names=["ag_news", "imdb"]):
    """
    Select datasets based on their names

    :param names: list of string of dataset names
    :return: list of dataset object
    """

    datasets = []

    if 'ag_news' in names:
        datasets.append(AgNews())
    if 'db_pedia' in names:
        datasets.append(DbPedia())
    if 'yelp_review' in names:
        datasets.append(YelpReview())
    if 'yelp_polarity' in names:
        datasets.append(YelpPolarity())
    if 'amazon_review' in names:
        datasets.append(AmazonReview())
    if 'amazon_polarity' in names:
        datasets.append(AmazonPolarity())
    if 'sogu_news' in names:
        datasets.append(SoguNews())
    if 'yahoo_answer' in names:
        datasets.append(YahooAnswer())
    if 'imdb' in names:
        datasets.append(Imdb())

    return datasets


if __name__ == "__main__":

    names = [
        'ag_news',
        'db_pedia',
        'yelp_review',
        'yelp_polarity',
        'amazon_review',
        'amazon_polarity',
        'sogu_news',
        'yahoo_answer',
        'imdb',
    ]

    for name in names:
        print("name: {}".format(name))
        dataset = load_datasets(names=[name])[0]
        # train data generator
        tr_gen = dataset.load_train_data(chunk_size=512)
        tr_sentences, labels = [], []
        for x, y in tr_gen:
            tr_sentences.extend(x)
            labels.extend(y)
        print(" train: (samples/labels) = ({}/{})".format(len(tr_sentences), len(labels)))

        # test data generator
        tr_gen = dataset.load_test_data(chunk_size=2048)
        te_sentences, labels = [], []
        for x, y in tr_gen:
            te_sentences.extend(x)
            labels.extend(y)
        print(" test: (samples/labels) = ({}/{})".format(len(te_sentences), len(labels)))


