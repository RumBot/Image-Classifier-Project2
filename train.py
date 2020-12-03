# Formatting checked in Black Playground
# Use the following command in the terminal to execute:
# python train.py

"""
@author: Bradley Davis
@title: Image Classifier training file
"""
# ---------------------------------------------------------------------------- #
# I like larger comment blocks like this, which aren't technically docstrings
# We learned about docstrings and all that in a previous lesson, but I didn't
# practice that much
# Need to bring in required libraries, first:
# argparse is useful for collecting and organizing input arguments for easy use
# torch is a required library for this module
# collections has cool functions for organizing data, like ordered
#   dictionaries
# os is general operating system stuff, bring in only what's needed
# If torch is imported as a whole, is it necessary to then import specific items
#   from torch individually? Probably just easier to keep track of what is being
#   used later
# ---------------------------------------------------------------------------- #

import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session

# ---------------------------------------------------------------------------- #
# Function definitions used in module:
# ---------------------------------------------------------------------------- #
# arg_parser() parses keywords as arguments from the command line
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    print("Parser has been defined.")
    # Add architecture selection to parser
    parser.add_argument(
        "--arch",
        type=str,
        help="Choose architecture from torchvision.models as str",
    )
    print("Architecture defined.")
    # Add checkpoint save directory to parser
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Define save directory for checkpoints as str. If not specified then model will be lost.",
    )
    print("Save directory defined.")

    # Add hyperparameter tuning to parser.
    # This is essentially a focus leveller or similar.
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Define gradient descent learning rate as float",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        help="Hidden units for DNN classifier as int",
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs for training as int"
    )
    print("Hyper kids have been added.")
    # Add GPU Option to parser
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU + Cuda for calculations"
    )
    print("GPU option has been defined.")
    # Parse args
    args = parser.parse_args()
    
    print("Return args")
    return args


# train_transformer(train_dir) performs training on a dataset.
# Transforms are like mutations and mating from MOEA / GA optimization.
def train_transformer(train_dir):
    # Define transformation
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # Load the Data
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data


# test_transformer(test_dir) performs test on a dataset.
# This would be like code body for evaluating fitness criteria in MOEA / GA.
def test_transformer(test_dir):
    # Define transformation
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # Load the Data
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data


# data_loader(data, train=True) creates a dataloader from dataset imported.
# Simple manual labor needed.
def data_loader(data, train=True):
    if train:
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader


# check_gpu(gpu_arg), use CUDA with GPU or CPU. GPU is faster.
def check_gpu(gpu_arg):
    # If gpu_arg is false simply return the cpu device
    if not gpu_arg:
        return torch.device("cpu")
    # If gpu_arg make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device


# primaryloader_model(architecture="vgg16")
# downloads model (primary) from torchvision
def primaryloader_model(architecture="vgg16"):
    # Load Defaults if none specified
    if type(architecture) == type(None):
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    else:
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    # Freeze parameters to avoid backprop
    for param in model.parameters():
        param.requires_grad = False
    return model


# initial_classifier(model, hidden_units) creates a classifier
# with the corect number of input layers
# I leaned on some examples for this section
# and went down the rabbit hole on YouTube about layers...
def initial_classifier(model, hidden_units):
    # Check that hidden layers has been input
    if type(hidden_units) == type(None):
        hidden_units = 4096  # hyperparamters
        print("Number of Hidden Layers specificed as 4096.")
    # Find Input Layers
    input_features = model.classifier[0].in_features

    # Define Classifier
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(input_features, hidden_units, bias=True)),
                ("relu1", nn.ReLU()),
                ("dropout1", nn.Dropout(p=0.5)),
                ("fc2", nn.Linear(hidden_units, 102, bias=True)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )
    return classifier


# validation(model, testloader, criterion, device) validates training
# against testloader to return loss and accuracy
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0

    for ii, (inputs, labels) in enumerate(testloader):

        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = labels.data == ps.max(dim=1)[1]
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


# network_trainer represents training of the network model
def network_trainer(model, trainloader, testloader, validloader, device, 
                  criterion, optimizer, epochs, print_every, steps):
    # Check Model Kwarg
    if type(epochs) == type(None):
        epochs = 5
        print("Number of epochs specificed as 5.")    
 
    print("Training process initializing .....\n")

    # Train Model
    for e in range(epochs):
        running_loss = 0
        model.train() # leave this in
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(testloader)))
            
                running_loss = 0
                model.train()

    return model


# validate_model(Model, Testloader, Device)
# validate the above model on test data images
def validate_model(model, testloader, device):
    # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        "Accuracy achieved by the network on test images is: %d%%"
        % (100 * correct / total)
    )


# initial_checkpoint(model, save_dir, train_data) saves
# the model at a defined checkpoint
def initial_checkpoint(model, save_dir, train_data):

    # Save model at checkpoint
    if type(save_dir) == type(None):
        save_dir = "/home/workspace/ImageClassifier/saved_models"
        print(save_dir)
    else:
        if isdir(save_dir):
            # Create `class_to_idx` attribute in model.
            # This doesn't work if I leave and start here later.
            model.class_to_idx = train_data.class_to_idx

            # Create checkpoint dictionary
            checkpoint = {
                "architecture": model.name,
                "classifier": model.classifier,
                "class_to_idx": model.class_to_idx,
                "state_dict": model.state_dict(),
            }

            # Save checkpoint
            torch.save(checkpoint, "my_checkpoint.pth")
        else:
            print("Directory not found, model will not be saved.")


# =============================================================================
# Main Function
# =============================================================================

# main() is where all the functions are called
def main():

    with active_session():
    # do long-running work here
        
        # Make Keyword Args for Training
        args = arg_parser()
        print("args returned...")
        print(args.save_dir)
        
        # Set directory for training
        data_dir = "flowers"
        train_dir = data_dir + "/train"
        valid_dir = data_dir + "/valid"
        test_dir = data_dir + "/test"

        # Pass transforms in, then create trainloader
        train_data = test_transformer(train_dir)
        valid_data = train_transformer(valid_dir)
        test_data = train_transformer(test_dir)

        trainloader = data_loader(train_data)
        validloader = data_loader(valid_data, train=False)
        testloader = data_loader(test_data, train=False)

        # Load Model
        model = primaryloader_model(architecture=args.arch)

        # Build Classifier
        model.classifier = initial_classifier(model, hidden_units=args.hidden_units)

        # Check for GPU
        device = check_gpu(gpu_arg=args.gpu)

        # Send model to device
        model.to(device)

        # Check for learnrate args
        if type(args.learning_rate) == type(None):
            learning_rate = 0.001
            print("Learning rate specificed as 0.001")
        else:
            learning_rate = args.learning_rate

        # Define loss and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

        # Define deep learning method
        print_every = 30
        steps = 0

        # Train the classifier layers using backpropogation
        trained_model = network_trainer(
            model,
            trainloader,
            testloader,
            validloader,
            device,
            criterion,
            optimizer,
            args.epochs,
            print_every,
            steps,)

        print("\nTraining process is now complete!!")

        # Quickly Validate the model
        validate_model(trained_model, testloader, device)

        # Save the model
        args.save_dir = "/home/workspace/ImageClassifier/saved_models"
        initial_checkpoint(trained_model, args.save_dir, train_data)


# =============================================================================
# Run Program
# =============================================================================
if __name__ == "__main__":
    main()
