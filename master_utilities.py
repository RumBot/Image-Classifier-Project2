# Train, Validate, Test
# Much help from:
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
# https://github.com/chauhan-nitin/Udacity-ImageClassifier
# https://github.com/jkielbaey/mlnd-image-classifier
# https://github.com/fotisk07/Image-Classifier

import os
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
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms

arch = {"vgg16":25088,"densenet121":1024,"alexnet":9216}

print("Master Utilities Ready...")

# directory needs to be updated because derpy derp mc_derp
# Not really needed because the directory has to be set in the command line
# before it'll find the file to run
base_dir = os.chdir("/home/workspace/ImageClassifier")

# New Function Definition
def load_data(flower_pot):
    '''
    Arguments : the data's path, pre-set in training module using argparse
    Returns : dataloaders, model
    
    Grab the directories for the train, validate, and test files and set transforms
    '''
    
    data_dir = flower_pot
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    print(data_dir)
    print(train_dir)
    print(valid_dir)
    print(test_dir)
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    dirs = {'train': train_dir, 
            'valid': valid_dir, 
            'test' : test_dir}
    
    image_datasets = {x: datasets.ImageFolder(dirs[x], transform = data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 32, shuffle = True) for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) 
                                  for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    
    with open('cat_to_name.json', 'r') as f:
        label_map = json.load(f)
    
    print("Data Loaders Ready...")
    
    return dataloaders

# New Function Definition
def nn_setup(structure = 'densenet121', dropout = 0.5, hidden_layer1 = 102, lr = 0.001, device = 'cpu'):
    '''
    Arguments:  The architecture for the network(alexnet,densenet121,vgg16)
                Hyperparameters for the network (hidden layer 1 nodes, dropout and learning rate)
                whether to use gpu or not
    Returns:    Optimizer, Criterion
    '''
    # Can use any model, just need to specify it in the function input
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("{} is not a valid model. Enter vgg16,densenet121,or alexnet.".format(structure))
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(arch[structure], 4096)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(4096, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
        
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr )

    print("Neural Net Ready...")
    
    return model, criterion, optimizer

# New Function Definition
def train_network(model, criterion, optimizer, scheduler, dataloaders, epochs = 5, device = 'cpu'):
    '''
    Arguments: The model, the criterion, the optimizer, the number of epochs, the dataset, and whether to use a gpu or not
    Returns: Nothing
    This function trains the model
    '''
    # Add some info regarding elapsed time
    since = time.time()
    
    if torch.cuda.is_available() and device == 'cuda':
        model.cuda()
        
    # Used to report best accuracy later
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    print("-------- Training Begin -------- ")
    
    for e in range(epochs):
        print('Epoch {}/{}'.format(e + 1, epochs))

        # Each epoch has a training and validation phase
        # This is cleaner than trying to call them individually
        for phase in ['train', 'valid']:
            if phase == 'train':
                optimizer.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        
        return model


    print("-------------- Training complete -----------------------")
    print("-------------- Have a nice day!! -----------------------")

# New Function Definition
def save_checkpoint(path = 'checkpoint.pth', structure = 'densenet121'):
    '''
    Arguments: Save path, structure, data sets
    Returns: Nothing
    This function saves the model at the user path
    '''
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    torch.save({'arch': structure,
                'state_dict': model.state_dict(), 
                'class_to_idx': model.class_to_idx}, 
                path)

    print("Checkpoint Save Complete... we hope!")
    
# New Function Definition    
def load_checkpoint(path = 'checkpoint.pth'):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases
    '''
    checkpoint = torch.load(path, map_location = torch.device('cpu'))
    
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr = checkpoint['lr']
    
    if structure == 'vgg16':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("{} is not a valid model. Enter vgg16,densenet121,or alexnet.".format(structure))
    
    # Create the classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(arch[structure], 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    # Put the classifier on the pretrained network
    model.classifier = classifier

    model.load_state_dict(checkpoint['state_dict'])

    calc_accuracy(model, 'test', True)
    
    print("Checkpoint Load Complete...")
    return model

# New Function Definition
def calc_accuracy(model, data, cuda=False):
    '''
    Arguments: Model and data
    Returns: nothing, printed output
    '''
    model.eval()
    model.to(device='cuda')    
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[data]):
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # obtain the outputs from the model
            outputs = model.forward(inputs)
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1)
            # check the 
            if idx == 0:
                print(predicted) #the predicted class
                print(torch.exp(_)) # the predicted probability
            equals = predicted == labels.data
            if idx == 0:
                print(equals)
            print(equals.float().mean())

            print("Accuracy Calculation Complete...")
    
# New Function Definition
def process_image(image_path):
    '''
    Arguments: The image's path
    Returns: The image as a tensor
    This function opens the image usign the PIL package, applies the  necessery transformations and returns the image as a tensor ready to be fed to the network
    '''

    for i in image_path:
        path = str(i)
    img = Image.open(i) # Here we open the image
    
    # Here as we did with the training data we will define a set of
    # transfomations that we will apply to the PIL image
    make_img_good = transforms.Compose([ 
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.transpose((2, 0, 1))
    ])

    tensor_image = make_img_good(img)

    print("Image Processing Complete...")
    
    return tensor_image

# New Function Definition
def imshow(tensor_image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is first
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    print("Open your mouth and close your eyes, here comes a plotted surprise!")
    
    return ax

# New Function Definition
def predict(image_path, model, topk = 5, device = 'cpu'):
    '''
    Arguments: The path to the image, the model, the number of prefictions and whether cuda will be used or not
    Returns: The "topk" most probable choices that the network predicts
    '''

    if torch.cuda.is_available() and device == 'gpu':
        model.to('cuda:0')

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img_torch)

    probability = F.softmax(output.data, dim = 1)

    print("The Great PyDini has made a prediction!")
    
    return probability.topk(topk)
