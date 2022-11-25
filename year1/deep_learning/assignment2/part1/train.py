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
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # Randomly initialize and modify the model's last layer for CIFAR100.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes, bias=False)
    nn.init.normal_(model.fc.weight, mean = 0.0, std=0.01)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_data, val_data = get_train_validation_set(data_dir)
    train_dl = data.DataLoader(train_data, batch_size)
    val_dl = data.DataLoader(val_data, batch_size)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with validation after each epoch. Save the best model.
    best_acc = 0
    model_path = data_dir + "/model"
    for epoch in range(epochs):

        print(f'Epoch {epoch+1}/{epochs}')
        print('-'*10)

        # training

        model.train()

        running_loss = 0.0
        running_corrects = 0

        for mini_batch in train_dl:
            inputs, labels = mini_batch
            inputs = inputs.to(device)

            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = nn.CrossEntropyLoss()
            l = loss(outputs, labels)

            # backward pass
            l.backward()
            optimizer.step()
            
            # stats
            running_loss += l.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_data)
        epoch_acc = running_corrects.double() / len(train_data)

        # validation
        val_acc = evaluate_model(model, val_dl, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Val Acc: {val_acc:.4f}')


    # Load the best model on val accuracy and return it.
    model.load_state_dict(torch.load(model_path))

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    running_corrects = 0
    total = 0

    for mini_batch in data_loader:
        with torch.no_grad():
            inputs, labels = mini_batch
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += len(labels)
        
        # stats
        running_corrects += torch.sum(preds == labels.data)

    accuracy = running_corrects.double() / total

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = get_model()

    # Get the augmentation to use

    # Train the model
    # best_model = train_model(model, lr, batch_size, epochs, data_dir, "checkpoint_name", device)
    model_path = data_dir + "/model"
    model.load_state_dict(torch.load(model_path))
    # Evaluate the model on the test set
    test_data = get_test_set(data_dir)
    test_dl = data.DataLoader(test_data, batch_size)
    
    test_acc = evaluate_model(model, test_dl, device)
    print(f'Test accuracy: {test_acc:.4f}')

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
