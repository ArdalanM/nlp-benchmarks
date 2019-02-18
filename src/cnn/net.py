# -*- coding: utf-8 -*-
"""
@author: DIAS Charles-Emmanuel <Charles-Emmanuel.Dias@lip6.fr>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_classes=2, input_length=1014, input_dim=68,
                 n_conv_filters=256,
                 n_fc_neurons=1024):
        super(CNN, self).__init__()

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

