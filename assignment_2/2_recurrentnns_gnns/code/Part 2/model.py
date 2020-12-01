# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.distributions
import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0',
                 embedding_dim=32, dropout=0):

        super(TextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_num_hidden, lstm_num_layers, dropout=dropout)
        self.out_linear = nn.Linear(lstm_num_hidden, vocabulary_size)

        self.vocabulary_size = vocabulary_size
        self.num_layers = lstm_num_layers
        self.num_hidden = lstm_num_hidden
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)

    def forward(self, x):
        out = self.embedding(x.long())
        out, _ = self.lstm(out)
        out = self.out_linear(out)
        return out
        #return torch.log_softmax(out, dim=-1)

    def sample(self, num_samples, length, temperature=-1, prompt=None):
        # note that the "temperature" parameter is really an inverse temperature,
        # with 0 meaning uniform sampling
        # negative inverse temperature is interpreted as greedy sampling
        hidden = torch.zeros(self.num_layers, num_samples, self.num_hidden).to(self.device)
        cell = torch.zeros(self.num_layers, num_samples, self.num_hidden).to(self.device)
        out = torch.empty(length, num_samples).to(self.device)
        if prompt == None:
            # start with a random letter
            out[0] = torch.randint(self.vocabulary_size, (num_samples, )).to(self.device)
            start = 1
        else:
            start = prompt.size(0)
            out[:start] = prompt[:, None]

        for i in range(1, length):
            x = self.embedding(out[i - 1].long())
            x, (hidden, cell) = self.lstm(x.unsqueeze(0), (hidden, cell))
            x = self.out_linear(x.squeeze())
            # don't overwrite prompt letters
            if i < start:
                continue
            if temperature < 0:
                out[i] = torch.argmax(x, dim=-1)
            else:
                probs = torch.softmax(temperature * x, dim=-1)
                dist = torch.distributions.Categorical(probs)
                out[i] = dist.sample((1, )).squeeze()
        return out.T.int()
