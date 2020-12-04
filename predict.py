import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import master_utilities

#Command Line Arguments
ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input_img', default='C:\Users\bdavis\Udacity_Work\Image_Classifier_Project2/flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='C:\Users\bdavis\Udacity_Work\Image_Classifier_Project2/saved_models/checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

parsedargs = ap.parse_args()
path_image = parsedargs.input_img
number_of_outputs = parsedargs.top_k
power = parsedargs.gpu
input_img = parsedargs.input_img
path = parsedargs.checkpoint

training_loader, testing_loader, validation_loader = master_utilities.load_data()

master_utilities.load_checkpoint(path)

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)

probabilities = master_utilities.predict(path_image, model, number_of_outputs, power)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])

i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

print("Now go away or I shall taunt you a second time!")
