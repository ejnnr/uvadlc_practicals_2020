###############################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Adapted: 2020-11-09
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import statistics

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# datasets
import datasets

# models
from bi_lstm import biLSTM
from lstm import LSTM
from gru import GRU
from peep_lstm import peepLSTM

import numpy as np

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

###############################################################################


def train(config, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Initialize the device which to run the model on
    device = torch.device(config.device)
    print(device)

    # Load dataset
    if config.dataset == 'randomcomb':
        print('Load random combinations dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = config.input_length
        dataset = datasets.RandomCombinationsDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

    elif config.dataset == 'bss':
        print('Load bss dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = 2
        config.input_dim = 3
        dataset = datasets.BaumSweetSequenceDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

        config.input_length = 4 * config.input_length

    elif config.dataset == 'bipalindrome':
        print('Load binary palindrome dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = 2
        dataset = datasets.BinaryPalindromeDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

        config.input_length = config.input_length*4+2-1



    # Setup the model that we are going to use
    if config.model_type == 'LSTM':
        print("Initializing LSTM model ...")
        model = LSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'biLSTM':
        print("Initializing bidirectional LSTM model...")
        model = biLSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'GRU':
        print("Initializing GRU model ...")
        model = GRU(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'peepLSTM':
        print("Initializing peephole LSTM model ...")
        model = peepLSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    # Setup the loss and optimizer
    loss_function = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    losses = []
    train_accuracies = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Move to GPU
        batch_inputs = batch_inputs.to(device)     # [batch_size, seq_length,1]
        batch_targets = batch_targets.to(device)   # [batch_size]

        # Reset for next iteration
        model.zero_grad()

        # Forward pass
        log_probs = model(batch_inputs)

        # Compute the loss, gradients and update network parameters
        loss = loss_function(log_probs, batch_targets)
        loss.backward()
        losses.append(loss.item())

        #######################################################################
        # Check for yourself: what happens here and why?
        #######################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)
        #######################################################################

        optimizer.step()

        predictions = torch.argmax(log_probs, dim=1)
        correct = (predictions == batch_targets).sum().item()
        accuracy = correct / log_probs.size(0)
        train_accuracies.append(accuracy)

        # print(predictions[0, ...], batch_targets[0, ...])

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 60 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                   Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        # Check if training is finished
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report
            # https://github.com/pytorch/pytorch/pull/9655
            break

        # Stop early if the last 100 losses were all low enough
        if all(x < 0.001 for x in losses[-100:]):
               break

    print('Done training.')

    # evaluate the model on new random data
    model.eval()
    test_accuracies = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Move to GPU
        batch_inputs = batch_inputs.to(device)     # [batch_size, seq_length,1]
        batch_targets = batch_targets.to(device)   # [batch_size]

        # Forward pass
        with torch.no_grad():
            log_probs = model(batch_inputs)
            predictions = torch.argmax(log_probs, dim=1)
            correct = (predictions == batch_targets).sum().item()
            accuracy = correct / log_probs.size(0)
        test_accuracies.append(accuracy)

        if step >= 5000 / config.batch_size:
            # If you receive a PyTorch data-loader error, check this bug report
            # https://github.com/pytorch/pytorch/pull/9655
            break

    return losses, train_accuracies, torch.tensor(test_accuracies).mean().item()
    ###########################################################################
    ###########################################################################


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='bipalindrome',
                        choices=['randomcomb', 'bss', 'bipalindrome'],
                        help='Dataset to be trained on.')
    # Model params
    parser.add_argument('--model_type', type=str, default='LSTM',
                        choices=['LSTM', 'biLSTM', 'GRU', 'peepLSTM'],
                        help='Model type: LSTM, biLSTM, GRU or peepLSTM')
    parser.add_argument('--input_length', type=int, default=6,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=256,
                        help='Number of hidden units in the model')
    # For binary input sequences (plus 0 padding), more than 3 is overkill
    parser.add_argument('--embedding_dim', type=int, default=3,
                        help='Embedding dimension')

    # Training params
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=3000,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    # Misc params
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5,
                        help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')


    # Train the model
    all_losses = []
    all_accs = []
    all_test_accs = []
    for seed in range(3):
        config = parser.parse_args()
        print(config)
        losses, train_accs, test_acc = train(config, seed)
        all_losses.append(losses)
        all_accs.append(train_accs)
        all_test_accs.append(test_acc)
    print("Test accuracies:", all_test_accs)
    print("Mean:", statistics.mean(all_test_accs), "+-", statistics.stdev(all_test_accs))

    for losses in all_losses:
        plt.plot(losses, color="blue", alpha=0.3)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("{}_losses_{}.pdf".format(config.model_type, config.input_length))
    plt.close()

    for accs in all_accs:
        plt.plot(accs, color="blue", alpha=0.3)
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.savefig("{}_train_accs_{}.pdf".format(config.model_type, config.input_length))
    plt.close()
    
