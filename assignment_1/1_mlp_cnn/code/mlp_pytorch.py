"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """
    
    def __init__(self, n_inputs, n_hidden, n_classes, better_init=False):
        """
        Initializes MLP object.
        
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
    
        TODO:
        Implement initialization of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()
        modules = []
        dims = list(zip([n_inputs] + n_hidden, n_hidden + [n_classes]))
        for i, (in_features, out_features) in enumerate(dims):
            linear = nn.Linear(in_features, out_features)
            modules.append(linear)
            if i < len(dims) - 1:
                #modules.append(nn.BatchNorm1d(out_features))
                modules.append(nn.ELU())
        modules.append(nn.Softmax(dim=1))

        # use ModuleList to register the submodules, necessary
        # to make things like MLP.parameters() etc. work
        self.layers = nn.ModuleList(modules)
        print(modules)

        def init_weights(m):
            if type(m) == nn.Linear:
                m.bias.data.fill_(0)
                nn.init.normal_(m.weight.data, 0, 0.0001)
        if not better_init:
            self.apply(init_weights)
        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out = x
        for module in self.layers:
            out = module(out)
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
