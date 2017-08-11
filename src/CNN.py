# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
import h5py
import argparse
import numpy as np
from sklearn import utils

import torch
import torch.nn as nn
from torch.autograd import Variable

from src import lib
from src.datasets import load_datasets
from src.lib import create_dataset

def get_args():
    parser = argparse.ArgumentParser(
        """Character-level convolutional networks for text classification: https://arxiv.org/abs/1509.01626""")
    parser.add_argument("--dataset", type=str, default='imdb')
    parser.add_argument("--model_folder", type=str, default="models/CNN/imdb")
    parser.add_argument("--alphabet", type=str, default="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    parser.add_argument("--maxlen", type=int, default=1014)
    parser.add_argument('--shuffle', action='store_true', default=False, help="shuffle train and test sets")
    parser.add_argument('--small_config', action='store_true', default=True, help='set conv feature map to 256 and fc neurons to 1024')
    parser.add_argument('--big_config', action='store_true', default=False, help='set conv feature map to 1024 and fc neurons to 2048')
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512, help="number of example read by the gpu during test time")
    parser.add_argument("--test_interval", type=int, default=50, help="Number of iterations between testing phases")
    parser.add_argument("--class_weights", nargs='+', type=float, default=None)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_halve_interval", type=float, default=100, help="Number of iterations before halving learning rate")
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    return args


def store_dataset(sentences, labels, vectorizer, filepath, chunk_size):

    f = h5py.File(filepath, 'w', libver='latest')
    id = 0

    for i in range(0, len(sentences), chunk_size):
        batch_data = vectorizer.transform(sentences[i: i+chunk_size])
        batch_label = np.array(labels[i: i+chunk_size])

        data_shape = (batch_data.shape[0], batch_data.shape[1], batch_data.shape[2])
        label_shape = (batch_label.shape[0],)

        xtr = f.create_dataset('data_{}'.format(id), shape=data_shape, dtype='f', compression='lzf')
        ytr = f.create_dataset('label_{}'.format(id), shape=label_shape, dtype='i8', compression='lzf')

        xtr[:] = batch_data
        ytr[:] = batch_label
        id += 1
    f.close()


def batchify(filename, batch_size=128):

    f = h5py.File(filename, 'r', libver='latest')
    n_chunks = len(f.keys()) // 2

    for i in range(n_chunks):
        x = f["data_{}".format(i)]
        y = f["label_{}".format(i)]
        for j in range(0, len(x), batch_size):
            yield x[j: j + batch_size], y[j: j+batch_size]
    f.close()


def predict_from_model(path, model, gpu=True):
    model.eval()
    y_prob = []

    generator = batchify(path, batch_size=opt.batch_size)

    for data in generator:
        tdata = [Variable(torch.from_numpy(x), volatile=True) for x in data]
        if gpu:
            tdata = [x.cuda() for x in tdata]

        yhat = model(tdata[0])

        # normalizing probs
        yhat = nn.functional.softmax(yhat)

        y_prob.append(yhat)

    y_prob = torch.cat(y_prob, 0)
    y_prob = y_prob.cpu().data.numpy()

    model.train()
    return y_prob


class create_model(nn.Module):
    def __init__(self, n_classes=2, input_length=1014, input_dim=68,
                 n_conv_filters=256,
                 n_fc_neurons=1024):
        super(create_model, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv1d(input_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(), nn.MaxPool1d(3))
        self.layer2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(), nn.MaxPool1d(3))
        self.layer3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(), nn.MaxPool1d(3))

        # layer 6 output length = (input_length - 96) / 27
        last_cnn_layer_len = (input_length - 96) / 27
        dim = int(last_cnn_layer_len * n_conv_filters)
        self.layer7 = nn.Sequential(nn.Linear(dim, n_fc_neurons), nn.Dropout(0.5))
        self.layer8 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(0.5))
        self.layer9 = nn.Linear(n_fc_neurons, n_classes)

        if n_conv_filters == 256 and n_fc_neurons == 1024:
            self.__init_weights(mean=0.0, std=0.05)
        elif n_conv_filters == 1024 and n_fc_neurons == 2048:
            self.__init_weights(mean=0.0, std=0.02)

    def __init_weights(self, mean=0.0, std=0.05):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(mean, std)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)

    def forward(self, x):

        x = x.transpose(1, 2)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        out = out.view(out.size(0), -1)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        return out


if __name__ == "__main__":

    opt = get_args()

    if not os.path.exists(opt.model_folder):
        os.makedirs(opt.model_folder)

    logger = lib.get_logger(logdir=opt.model_folder, logname="logs.txt")
    logger.info("parameters: {}".format(vars(opt)))

    dataset = load_datasets(names=[opt.dataset])[0]
    dataset_name = dataset.__class__.__name__
    n_classes = dataset.n_classes

    logger.info("dataset: {}, n_classes: {}".format(dataset_name, n_classes))

    tr_path = "{}/{}_train.h5".format(opt.model_folder, dataset_name)
    te_path = "{}/{}_test.h5".format(opt.model_folder, dataset_name)

    if os.path.exists(tr_path) and os.path.exists(te_path):
        logger.info("loading from:\n-{}\n-{}".format(tr_path, te_path))
    else:
        logger.info("creating dataset:")

        logger.info("  - loading dataset...")
        tr_data = dataset.load_train_data(chunk_size=opt.chunk_size)
        te_data = dataset.load_test_data(chunk_size=opt.chunk_size)

        logger.info("  - loading train samples...")
        tr_sentences, tr_labels = create_dataset(tr_data)
        logger.info("  - loading train samples... {} samples".format(len(tr_sentences)))

        logger.info("  - loading test samples...")
        te_sentences, te_labels = create_dataset(te_data)
        logger.info("  - loading test samples... {} samples".format(len(tr_sentences)))

        logger.info("  - shuffling...")
        if opt.shuffle:
            tr_sentences, tr_labels = utils.shuffle(tr_sentences, tr_labels, random_state=opt.seed)
            te_sentences, te_labels = utils.shuffle(te_sentences, te_labels, random_state=opt.seed)

        vectorizer = lib.CharOHEncoder(alphabet=opt.alphabet, maxlen=opt.maxlen, dtype=np.float16)
        vectorizer.fit(tr_sentences)

        logger.info("  - storing train set: {}".format(tr_path))
        store_dataset(tr_sentences, tr_labels, vectorizer, tr_path, opt.chunk_size)

        logger.info("  - storing test set: {}".format(te_path))
        store_dataset(te_sentences, te_labels, vectorizer, te_path, opt.chunk_size)
        logger.info("Creating dataset: done.")

    logger.info("model:")
    logger.info("  - calculate number of batches...")
    tr_generator = batchify(tr_path, batch_size=opt.batch_size)
    te_generator = batchify(te_path, batch_size=opt.batch_size)
    yte = np.concatenate([y for x, y in te_generator])

    tr_iteration = len([x.shape[0] for x, _ in tr_generator])
    te_iteration = len([x.shape[0] for x, _ in te_generator])
    logger.info("  - calculate number of batches: train: {}, test: {}".format(tr_iteration, te_iteration))

    if opt.small_config:
        logger.info("  - creating network (small config)")
        model = create_model(input_length=opt.maxlen, n_classes=n_classes, input_dim=len(opt.alphabet),
                             n_conv_filters=256, n_fc_neurons=1024)

    elif opt.big_config:
        logger.info("  - creating network (big config)")
        model = create_model(input_length=opt.maxlen, n_classes=n_classes, input_dim=len(opt.alphabet),
                             n_conv_filters=1024, n_fc_neurons=2048)
    else:
        raise

    if opt.gpu:
        model.cuda()

    if opt.class_weights:
        criterion = nn.CrossEntropyLoss(torch.cuda.FloatTensor(opt.class_weights))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    model.train()

    tr_gen = batchify(tr_path, batch_size=opt.batch_size)

    for n_iter in range(opt.iterations):
        try :
            data = tr_gen.__next__()
        except StopIteration:
            tr_gen = batchify(tr_path, batch_size=opt.batch_size)
            data = tr_gen.__next__()

        tdata = [Variable(torch.from_numpy(x)) for x in data]
        if opt.gpu:
            tdata = [x.cuda() for x in tdata]

        tx, ty_true = tdata
        y_true = data[-1]

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        yhat = model(tx)
        y_prob = yhat.cpu().data.numpy()

        loss = criterion(yhat, ty_true)
        loss.backward()
        optimizer.step()

        tr_metrics = lib.get_metrics(y_true, y_prob, n_classes=n_classes, list_metrics=['accuracy', 'log_loss'])

        params = [dataset_name, n_iter, opt.iterations, tr_metrics]
        logger.info('{} - Iter [{}/{}] - train metrics: {}'.format(*params))

        if n_iter % opt.test_interval == 0:

            y_prob = predict_from_model(te_path, model, gpu=opt.gpu)
            te_metrics = lib.get_metrics(yte, y_prob, n_classes=n_classes, list_metrics=['accuracy', 'log_loss'])
            params = [dataset_name, n_iter, opt.iterations, tr_metrics, te_metrics]
            logger.info('{} - Iter [{}/{}] - train metrics: {}, test metrics: {}'.format(*params))

            diclogs = {
                "predictions": {
                    "test": {
                        "y_true": yte,
                        "y_prob": y_prob
                    }
                },
                "name": "CNN",
                "parameters": vars(opt)
            }

            import pickle
            filename = "diclog_[{}|{}]_loss[{:.3f}|{:.3f}]_acc[{:.3f}|{:.3f}].pkl".format(n_iter, opt.iterations, tr_metrics['logloss'], te_metrics['logloss'],
                                                                         tr_metrics['accuracy'], te_metrics['accuracy'])

            with open('{}/{}'.format(opt.model_folder, filename), 'wb') as f:
                pickle.dump(diclogs, f, protocol=4)


        if n_iter % opt.lr_halve_interval == 0 and n_iter > 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            lr /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logger.info("new lr: {}".format(lr))
