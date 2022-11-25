################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    preds = predictions.argmax(axis=1)    

    conf_mat = np.zeros((10, 10))
    for K in range(10):
      for i in range(len(targets)):
        if preds[i] == K and targets[i] == K:
          conf_mat[preds[i]][targets[i]] += 1
        elif preds[i] != K and targets[i] == K:
          conf_mat[targets[i]][preds[i]] += 1

    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    metrics = dict()

    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix, axis=0) - TP
    FN = np.sum(confusion_matrix, axis=1) - TP
    TN = []
    for i in range(10):
      temp = np.delete(confusion_matrix, i, 0)
      temp = np.delete(temp, i, 1)
      TN.append(int(sum(sum(temp))))

    # calculate the metrics
    metrics['precision'] = np.array(TP/(TP+FP), dtype=np.float64)
    metrics['recall'] = np.array(TP/(TP + FN), dtype=np.float64)
    metrics['accuracy'] = np.array(np.sum(TP) / np.sum(confusion_matrix), dtype=np.float64)
    metrics['f1_beta'] = np.array((1 + beta**2) * metrics['precision'] * metrics['recall'] / 
                        (beta**2 * metrics['precision'] + metrics['recall']), dtype=np.float64)

    # remove nan values
    metrics['precision'] = np.nan_to_num(metrics['precision'])
    metrics['recall'] = np.nan_to_num(metrics['recall'])
    metrics['accuracy'] = np.nan_to_num(metrics['accuracy'])
    metrics['f1_beta'] = np.nan_to_num(metrics['f1_beta'])

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    precision = []
    recall = []
    accuracy = []
    f1_beta = []

    metrics = dict()

    for mini_batch in data_loader:
      inputs, labels = mini_batch
      inputs = inputs.flatten().reshape((inputs.shape[0], 3*32*32))
      output = model.forward(inputs)
      cm = confusion_matrix(output, labels)
      cm_metrics = confusion_matrix_to_metrics(cm)

      precision.append(cm_metrics['precision'])
      recall.append(cm_metrics['recall'])
      accuracy.append(cm_metrics['accuracy'])
      f1_beta.append(cm_metrics['f1_beta'])

    metrics['precision'] = np.mean(precision, axis = 0).tolist()
    metrics['recall'] = np.mean(recall, axis = 0).tolist()
    metrics['accuracy'] = np.mean(accuracy, axis = 0).tolist()
    metrics['f1_beta'] = np.mean(f1_beta, axis = 0).tolist()

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    model = MLP(3*32*32, hidden_dims, 10)
    loss_module = CrossEntropyModule()

    # TODO: Training loop including validation
    val_accuracies = []
    loss_progress = []
    best_acc = 0

    for epoch in range(epochs):

      epoch_loss = []

      for mini_batch in cifar10_loader['train']:

        inputs, labels = mini_batch
        inputs = inputs.flatten().reshape(batch_size, 3*32*32)

        forward = model.forward(inputs)
        loss = loss_module.forward(forward, labels)
        epoch_loss.append(loss)

        loss_grad = loss_module.backward(forward, labels)
        model.backward(loss_grad)

        for layer in model.layers:
          if hasattr(layer, 'params'):
            layer.params['weight'] -= lr * layer.grads['weight']
            layer.params['bias'] -= lr * layer.grads['bias']

      train_loss = np.mean(epoch_loss)
      loss_progress.append(train_loss)

      # validate model up to this epoch
      validation_metrics = evaluate_model(model, cifar10_loader['validation'])
      val_acc = np.mean(validation_metrics['accuracy'])
      val_accuracies.append(val_acc)

      # save best model
      if val_acc > best_acc:
        best_model = model
        best_acc = val_acc
      
      print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Acc: {:.6f}'.format(
        epoch+1, 
        train_loss,
        val_acc
        ))

    # TODO: Test best model
    model_evaluation = evaluate_model(best_model, cifar10_loader['test'])
    test_accuracy = np.mean(model_evaluation['accuracy'])
    print("Test Acc: ", test_accuracy)

    # get te confusion matrix of the test dataset
    conf_matrix = np.zeros((10, 10))
    for mini_batch in cifar10_loader['test']:
      inputs, labels = mini_batch
      inputs = inputs.flatten().reshape((inputs.shape[0], 3*32*32))
      output = best_model.forward(inputs)
      conf_matrix = conf_matrix + confusion_matrix(output, labels)

    # TODO: Add any information you might want to save for plotting
    logging_dict = {
      'loss_progress': loss_progress,
      'val_accuracies': val_accuracies,
      'conf_matrix': conf_matrix.tolist(),
      'evaluation_metrics': model_evaluation,
      'test_accuracy': test_accuracy
    }

    #######################
    # END OF YOUR CODE    #
    #######################

    return best_model, val_accuracies, test_accuracy, logging_dict

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    
