"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()

        def create_conv_block(channels):
            return (
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU()
            )

        class PreactBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv = nn.Sequential(*create_conv_block(channels))

            def forward(self, x):
                return x + self.conv(x)

        def create_preact_block(in_channels, out_channels):
            if in_channels == out_channels:
                return (
                    PreactBlock(in_channels),
                    PreactBlock(in_channels),
                    nn.MaxPool2d(3, stride=2, padding=1)
                )
            return (
                PreactBlock(in_channels),
                PreactBlock(in_channels),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.MaxPool2d(3, stride=2, padding=1)
            )
                

        self.sequential = nn.Sequential(
            # The first block is different than the rest
            # conv0
            nn.Conv2d(3, 64, 3, padding=1),
            # PreAct1
            PreactBlock(64),
            # conv1
            nn.Conv2d(64, 128, 1),
            # maxpool1
            nn.MaxPool2d(3, stride=2, padding=1),

            # Block 2
            *create_preact_block(128, 256),

            # Block 3
            *create_preact_block(256, 512),

            # Block 4
            *create_preact_block(512, 512),

            # Block 5
            *create_preact_block(512, 512),

            # Linear
            nn.Flatten(),
            nn.Linear(512, n_classes)
        )
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
        out = self.sequential(x)
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
