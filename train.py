import time
import json
import copy
import PIL
from PIL import Image
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch
from torch import nn, optim
from torch.optim import lr_scheduler as scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms

# Utility file for handling function calls
import master_utilities
# Don't die file for prevention of wasting GPU time again
from workspace_utils import active_session

# Set up the argument parser and load command line variables
args = argparse.ArgumentParser(description = 'train.py')
# Command Line ardguments

args.add_argument('data_dir', nargs = '*', action = "store", default = "/home/workspace/ImageClassifier/flowers")
args.add_argument('--device', dest = "device", action = "store", default = "cpu")
args.add_argument('--save_dir', dest = "save_dir", action = "store", default = "/checkpoint.pth")
args.add_argument('--learning_rate', dest = "learning_rate", action = "store", default = 0.001)
args.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
args.add_argument('--epochs', dest = "epochs", action = "store", type = int, default = 5)
args.add_argument('--arch', dest = "arch", action = "store", default = "densenet121", type = str)
args.add_argument('--hidden_units', dest = "hidden_units", action = "store",type = int,  default = 102)

# Collect and organize arguments
parsedargs = args.parse_args()
flower_pot = parsedargs.data_dir
path = parsedargs.save_dir
lr = parsedargs.learning_rate
structure = parsedargs.arch
dropout = parsedargs.dropout
hidden_layer1 = parsedargs.hidden_units
device = parsedargs.device
epochs = parsedargs.epochs

with active_session():
    # Generate the data loaders from master utilities
    # Input argument needed: directory where flower data is stored
    dataloaders = master_utilities.load_data(flower_pot)
    
    # Generate model within neural net setup routine
    # Return model, criterion, and optimizer
    # ORDER MATTERS!
    model, criterion, optimizer = master_utilities.nn_setup(structure, dropout, hidden_layer1, lr, device)

    # Begin training the neural net
    # Input the model, optimizer, criterion (ORDER MATTERS!!)
    # Input from parser: epochs, framework selection (cpu, gpu)
    # Input from user:  'print every = 20', dataloaders
    master_utilities.train_network(model, criterion, optimizer, scheduler, dataloaders, epochs, device)

    # Save the checkpoint data
    # Huge issues with this step for whatever reason
    master_utilities.save_checkpoint(path = 'checkpoint.pth', structure = 'densenet121')

print("Classifier has been trained")
