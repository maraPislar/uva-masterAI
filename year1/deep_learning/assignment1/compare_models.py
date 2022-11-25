################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch
import train_mlp_numpy

import torch
import torch.nn as nn
import torch.optim as optim
# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.
import json
import seaborn as sns
from matplotlib import pyplot as plt

def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    numpy_best_model_metrics, val_accuracies, test_accuracy, numpy_logging_dict = train_mlp_numpy.train(hidden_dims=[128],
                                                                                                        lr=0.1,
                                                                                                        batch_size=128,
                                                                                                        epochs=10,
                                                                                                        seed=42,
                                                                                                        data_dir='data/')

    # original model

    pytorch_best_model_metrics, val_accuracies, test_accuracy, pytorch_logging_dict = train_mlp_pytorch.train(hidden_dims=[128],
                                                                                                              lr=0.1,
                                                                                                              use_batch_norm=False,
                                                                                                              batch_size=128,
                                                                                                              epochs=10,
                                                                                                              seed=42,
                                                                                                              data_dir='data/')
    
    # beta metrics tests

    f1_beta_scores = [0.1, 1, 10]
    f1_beta_metrics = dict.fromkeys(f1_beta_scores, None)
    for f1_beta in f1_beta_scores:
        metrics = train_mlp_pytorch.confusion_matrix_to_metrics(pytorch_logging_dict['conf_matrix'], f1_beta)
        f1_beta_metrics[f1_beta] = metrics

    # 9 different learning rates tests

    learning_rates = np.geomspace(start=0.000001, stop=100, num=9, endpoint=True)
    lr_metrics = dict.fromkeys(learning_rates, None)
    for lr in learning_rates:
       model, val_accuracies, test_accuracy, logging_dict = train_mlp_pytorch.train(hidden_dims=[128],
                                                                                       lr=lr,
                                                                                       use_batch_norm=False,
                                                                                       batch_size=128,
                                                                                       epochs=10,
                                                                                       seed=42,
                                                                                       data_dir='data/')
       lr_metrics[lr] = logging_dict

    # # test model with different number of hidden layers (20 epochs)

    model, val_accuracies, test_accuracy, logging_dict_1 = train_mlp_pytorch.train(hidden_dims=[128],
                                                                                   lr=0.1,
                                                                                   use_batch_norm=False,
                                                                                   batch_size=128,
                                                                                   epochs=20,
                                                                                   seed=42,
                                                                                   data_dir='data/')

    model, val_accuracies, test_accuracy, logging_dict_2 = train_mlp_pytorch.train(hidden_dims=[256, 128],
                                                                                   lr=0.1,
                                                                                   use_batch_norm=False,
                                                                                   batch_size=128,
                                                                                   epochs=20,
                                                                                   seed=42,
                                                                                   data_dir='data/')

    model, val_accuracies, test_accuracy, logging_dict_3 = train_mlp_pytorch.train(hidden_dims=[512, 256, 128],
                                                                                   lr=0.1,
                                                                                   use_batch_norm=False,
                                                                                   batch_size=128,
                                                                                   epochs=20,
                                                                                   seed=42,
                                                                                   data_dir='data/')

    # TODO: Run all hyperparameter configurations as requested
    results = {
        'numpy_best_model_metrics': numpy_logging_dict,
        'pytorch_best_model_metrics': pytorch_logging_dict,
        'f1_beta_metrics': f1_beta_metrics,
        'lr_metrics': lr_metrics,
        '1hd_metrics': logging_dict_1,
        '2hd_metrics': logging_dict_2,
        '3hd_metrics': logging_dict_3
    }

    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file
    with open(results_filename, "w") as write_file:
        json.dump(results, write_file, indent=4)

    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    # open results file
    f = open(results_filename)
    results = json.load(f)

    """ 
        Numpy best model analysis 
    """

    # plot training loss
    best_model_losses = results['numpy_best_model_metrics']['loss_progress']
    plt.plot(range(1, 11), best_model_losses, label="loss")
    plt.title("Loss progress while training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("numpy_best_model_training_loss.png")
    plt.close()

    # plot validation accuracies
    best_model_val_accuracies = results['numpy_best_model_metrics']['val_accuracies']
    plt.plot(range(1, 11), best_model_val_accuracies, label="accuracy")
    plt.title("Validation accuracy while training")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("numpy_best_model_validation_accuracy.png")
    plt.close()

    """ 
        Pytorch best model analysis 
    """

    # plot training loss
    best_model_losses = results['pytorch_best_model_metrics']['loss_progress']
    plt.plot(range(1, 11), best_model_losses, label="loss")
    plt.title("Loss progress while training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("pytorch_best_model_training_loss.png")
    plt.close()

    # plot validation accuracies
    best_model_val_accuracies = results['pytorch_best_model_metrics']['val_accuracies']
    plt.plot(range(1, 11), best_model_val_accuracies, label="accuracy")
    plt.title("Validation accuracy while training")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("pytorch_best_model_validation_accuracy.png")
    plt.close()

    # confusion matrix
    conf_matrix = np.array(results['pytorch_best_model_metrics']['conf_matrix'])
    conf_matrix = np.reshape(conf_matrix, (10,10))

    labels = [i for i in range(10)]
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, square=False, fmt='g').set(title='Confusion matrix of test data')
    plt.savefig('conf_matrix_test_data.png')
    plt.close()

    """ 
        Learning rates
    """

    learning_rates = np.geomspace(start=0.000001, stop=100, num=9, endpoint=True)
    lr_metrics = dict.fromkeys(learning_rates, None)
    training_loss = []
    validation_accuracies = []
    for lr in lr_metrics:
      training_loss.append(results['lr_metrics'][str(lr)]['loss_progress'])
      validation_accuracies.append(results['lr_metrics'][str(lr)]['val_accuracies'])

    for i in range(len(training_loss)):
      plt.plot(training_loss[i], label='lr='+str(learning_rates[i]))
    plt.title('Training Losses for different learning rates')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('lr_training_losses.png')
    plt.close()

    for i in range(len(validation_accuracies)):
      plt.plot(validation_accuracies[i], label='lr='+str(learning_rates[i]))
    plt.title('Validation accuracies for different learning rates')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.savefig('lr_validation_accuracy.png')
    plt.close()

    """
        Different hidden layers
    """

    # 1 layer metrics
    hd1_loss = results['1hd_metrics']['loss_progress']
    hd1_acc = results['1hd_metrics']['val_accuracies']

    # 2 layer metrics
    hd2_loss = results['2hd_metrics']['loss_progress']
    hd2_acc = results['2hd_metrics']['val_accuracies']

    # 3 layer metrics
    hd3_loss = results['3hd_metrics']['loss_progress']
    hd3_acc = results['3hd_metrics']['val_accuracies']

    # plot the training loss for each model
    plt.plot(range(1, 21), hd1_loss, label='1 hidden layer')
    plt.plot(range(1, 21), hd2_loss, label='2 hidden layers')
    plt.plot(range(1, 21), hd3_loss, label='3 hidden layers')
    plt.title('Training loss for 1, 2, 3 hidden layers')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('different_hidden_layers_loss.png')
    plt.close()

    # plot the validation accuracy for each model
    plt.plot(range(1, 21), hd1_acc, label='1 hidden layer')
    plt.plot(range(1, 21), hd2_acc, label='2 hidden layers')
    plt.plot(range(1, 21), hd3_acc, label='3 hidden layers')
    plt.title('Validation accuracy for 1, 2, 3 hidden layers')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('different_hidden_layers_accuracy.png')
    plt.close()

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'results.json' 
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)