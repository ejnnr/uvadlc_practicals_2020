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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################


def train(config):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(os.path.join(config.summary_path, current_time))

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length,
                                dataset.vocab_size, config.lstm_num_hidden,
                                config.lstm_num_layers, device,
                                config.embedding_dim, 1 - config.dropout_keep_prob).to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          config.learning_rate_step,
                                          config.learning_rate_decay)

    step = 0
    # we want to be able to train multiple epochs to reach config.train_steps, so wrap the given
    # loop into another one.
    # step is increased manually
    while step < config.train_steps:
        for _, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            batch_inputs = torch.stack(batch_inputs).to(device)
            batch_targets = torch.stack(batch_targets).to(device)

            optimizer.zero_grad()

            # We have shape (seq_len, batch_size, vocab_size)
            # But nn.CrossEntropyLoss expects (batch_size, vocab_size, seq_len)
            logits = model(batch_inputs).permute(1, 2, 0)
            batch_targets = batch_targets.T

            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        max_norm=config.max_norm)

            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == batch_targets).float().mean().item()

            writer.add_scalar("Loss", loss.item(), step)
            writer.add_scalar("Accuracy", accuracy, step)
            writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], step)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if (step + 1) % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                        Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                        ))

            if (step + 1) % config.sample_every == 0:
                samples = model.sample(5, 100)
                print("Samples after {} steps:".format(step + 1))
                for sample in samples:
                    print(dataset.convert_to_string(sample.tolist()))
            if (step + 1) % config.save_every == 0:
                print("Saving model checkpoint at step {}".format(step))
                # we only save the model so we can later use it for sampling
                # Using this to resume training might have unexpected results because
                # scheduler and optimizer state are not saved!
                torch.save(model.state_dict(), "models/step_{}_{}.pth".format(step, time.time()))

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error,
                # check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
            step += 1

    print('Done training.')


###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Embedding dimension')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--save_every', type=int, default=10000,
                        help='How often so save the current model')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    train(config)
