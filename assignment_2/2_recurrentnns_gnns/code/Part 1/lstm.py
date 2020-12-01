"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_dim, out_dim, nonlinearity='sigmoid', bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.kaiming_normal_(self.W, nonlinearity=nonlinearity)
        if bias:
            self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        if hasattr(self, 'b'):
            return x @ self.W + self.b[None]
        else:
            return x @ self.W

class Gate(nn.Module):
    def __init__(self, input_dim, hidden_dim, nonlinearity=torch.sigmoid):
        super().__init__()
        # we only need one bias since the outputs are added
        self.hx = MyLinear(input_dim, hidden_dim, bias=False)
        self.hh = MyLinear(hidden_dim, hidden_dim, nonlinearity='tanh')
        self.nonlinearity = nonlinearity

    def forward(self, x, h):
        return self.nonlinearity(self.hx(x) + self.hh(h))

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device, embedding_dim=2):

        super(LSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # hard-coded number of embeddings for binary sequences + 0 for padding
        self.embedding = nn.Embedding(3, embedding_dim)
        self.gates = nn.ModuleDict()

        for key in "ifo":
            self.gates[key] = Gate(embedding_dim, hidden_dim)
        self.gates['g'] = Gate(embedding_dim, hidden_dim, nonlinearity=torch.tanh)

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
        self.h = torch.zeros(self.hidden_dim, self.batch_size).to(self.device)
        self.c = torch.zeros(self.hidden_dim, self.batch_size).to(self.device)
        for i in range(x.size(1)):
            embedded = self.embedding(x[:, i, 0].long())
            self.next_state(embedded)
            # reset hidden/cell state if we just saw a padding element
            # see @280 on Piazza
            # self.h[:, x[:, i, 0] == 0] = 0
            # self.c[:, x[:, i, 0] == 0] = 0
        p = self.linear_p(self.h)
        return torch.log_softmax(p, dim=-1)
        ########################
        # END OF YOUR CODE    #
        #######################

    def next_state(self, x):
        """Calculate next cell state and hidden state for a single time step."""
        act = {}
        for key, gate in self.gates.items():
            act[key] = gate(x, self.h)
        self.c = act['g'] * act['i'] + self.c * act['f']
        self.h = torch.tanh(self.c) * act['o']
