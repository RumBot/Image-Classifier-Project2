# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

# Utility file for handling function calls
import master_utilities

# Set up the argument parser and load command line variables
args = argparse.ArgumentParser(description='train.py')
# Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="cpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

# Collect and organize arguments
parsedargs = args.parse_args()
flowers = parsedargs.data_dir
workout = parsedargs.save_dir
lr = parsedargs.learning_rate
structure = parsedargs.arch
dropout = parsedargs.dropout
hidden_layer1 = parsedargs.hidden_units
power = parsedargs.gpu
epochs = parsedargs.epochs

trainloader, v_loader, testloader = master_utilities.load_data(flowers)

model, optimizer, criterion = master_utilities.nn_setup(structure,dropout,hidden_layer1,lr,power)

master_utilities.train_network(model, optimizer, criterion, epochs, 20, trainloader, power)

master_utilities.save_checkpoint(workout,structure,hidden_layer1,dropout,lr)

print("Classifier has been trained")
