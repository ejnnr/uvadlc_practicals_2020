"""
This module implements a LSTM with peephole connections in PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lstm import MyLinear, Gate


class peepLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device, embedding_dim=2):

        super(peepLSTM, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # hard-coded number of embeddings for binary sequences + 0 for padding
        self.embedding = nn.Embedding(3, embedding_dim)
        self.gates = nn.ModuleDict()

        for key in "ifo":
            self.gates[key] = Gate(embedding_dim, hidden_dim)

        self.linear_c = MyLinear(embedding_dim, hidden_dim)
        self.linear_p = MyLinear(hidden_dim, num_classes, nonlinearity='tanh')

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.device = device
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.c = torch.zeros(self.hidden_dim, self.batch_size).to(self.device)
        for i in range(x.size(1)):
            embedded = self.embedding(x[:, i, 0].long())
            h = self.next_state(embedded)
            # reset hidden/cell state if we just saw a padding element
            # see @280 on Piazza
            #self.c[:, x[:, i, 0] == 0] = 0
        p = self.linear_p(h)
        return torch.log_softmax(p, dim=-1)
        ########################
        # END OF YOUR CODE    #
        #######################

    def next_state(self, x):
        """Calculate next cell state for a single time step."""
        act = {}
        for key, gate in self.gates.items():
            act[key] = gate(x, self.c)
        self.c = self.linear_c(x) * act['i'] + self.c * act['f']
        return torch.tanh(self.c) * act['o']
