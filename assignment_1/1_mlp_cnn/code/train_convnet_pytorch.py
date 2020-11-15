"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    if torch.cuda.is_available():
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    net = ConvNet(3, 10).to(device)
    cifar = cifar10_utils.get_cifar10(FLAGS.data_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.learning_rate)
    # decay LR by 0.5 every 500 iterations
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    losses = []
    accuracies = []
    for step in range(FLAGS.max_steps):
        # load the next batch
        x, y = cifar["train"].next_batch(FLAGS.batch_size)
        x = torch.from_numpy(x).to(device)
        # Normalize per channel
        #x = x / torch.Tensor([62.245, 60.982, 65.468]).view(1, 3, 1, 1)
        #x = x.reshape(FLAGS.batch_size, -1)
        y = torch.from_numpy(y)
        y = torch.argmax(y, dim=1).to(device)

        # forward pass
        net.train()
        out = net(x)
        loss = criterion(out, y)
        losses.append(loss.item())

        # backward pass + weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

        # evaluate every FLAGS.eval_freq iterations
        if (step + 1) % FLAGS.eval_freq == 0:
            net.eval()
            # during test time, batch norm uses the running stats from training,
            # so the batch size doesn't matter and we can choose one which divides
            # the total number of samples
            num_batches = 100
            bs = 100
            mean_acc = 0
            for i in range(num_batches):
                x, y = cifar["train"].next_batch(bs)
                x = torch.from_numpy(x).to(device)
                y = torch.from_numpy(y).to(device)
                #x = x / torch.Tensor([62.245, 60.982, 65.468]).view(1, 3, 1, 1)
                with torch.no_grad():
                    out = net(x)
                    mean_acc += accuracy(out, y) / num_batches
            accuracies.append(mean_acc)
            print("Step {}, accuracy: {:.5f} %".format(step + 1, mean_acc * 100))
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Number of batches")
    plt.ylabel("Batch loss")
    plt.savefig("fig/loss_curve.pdf")
    plt.close()

    plt.figure()
    plt.plot(range(1, FLAGS.eval_freq * len(accuracies) + 1, FLAGS.eval_freq), accuracies)
    plt.xlabel("Number of batches")
    plt.ylabel("Accuracy on the test set")
    plt.savefig("fig/accuracy_curve.pdf")
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
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
