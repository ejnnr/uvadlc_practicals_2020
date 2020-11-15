"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    n = targets.size(0)
    numeric_predictions = torch.argmax(predictions, dim=1)
    accuracy = torch.sum(targets[torch.arange(n), numeric_predictions]) / n
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    mlp = MLP(3 * 32 * 32, dnn_hidden_units, 10, FLAGS.better_init)
    cifar = cifar10_utils.get_cifar10(FLAGS.data_dir)
    def criterion(pred, target):
        return nn.functional.nll_loss(pred.log(), target)
    optimizer = optim.Adam(mlp.parameters(), lr=FLAGS.learning_rate, weight_decay=0)
    # decay LR by 0.5 every 500 iterations
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    losses = []
    accuracies = []
    for step in range(FLAGS.max_steps):
        # load the next batch
        x, y = cifar["train"].next_batch(FLAGS.batch_size)
        x = torch.from_numpy(x)
        if FLAGS.better_init:
            # Normalize per channel
            x = x / torch.Tensor([62.245, 60.982, 65.468]).view(1, 3, 1, 1)
        x = x.reshape(FLAGS.batch_size, -1)
        y = torch.from_numpy(y)
        y = torch.argmax(y, dim=1)

        # forward pass
        out = mlp(x)
        loss = criterion(out, y)
        losses.append(loss.item())

        # backward pass + weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # evaluate every FLAGS.eval_freq iterations
        if (step + 1) % FLAGS.eval_freq == 0:
            x, y = cifar["test"].images, cifar["test"].labels
            # Normalize per channel
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            if FLAGS.better_init:
                x = x / torch.Tensor([62.245, 60.982, 65.468]).view(1, 3, 1, 1)
            x = x.reshape(10000, -1)
            with torch.no_grad():
                out = mlp(x)
                acc = accuracy(out, y)
                accuracies.append(acc)
                print("Step {}, accuracy: {:.5f} %".format(step + 1, acc * 100))
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Number of batches")
    plt.ylabel("Batch loss")
    plt.savefig("../fig/torch_mlp/loss_curve.pdf")
    plt.close()

    plt.figure()
    plt.plot(range(1, FLAGS.eval_freq * len(accuracies) + 1, FLAGS.eval_freq), accuracies)
    plt.xlabel("Number of batches")
    plt.ylabel("Accuracy on the test set")
    plt.savefig("../fig/torch_mlp/accuracy_curve.pdf")
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--better-init', dest="better_init", action="store_true",
                        help='Use default PyTorch initialization')
    parser.set_defaults(better_init=False)
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
