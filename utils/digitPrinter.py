import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import data
from torch.backends import cudnn
from torchvision import transforms

import pandas as pd
import numpy as np
import random as rd

import matplotlib.pyplot as plt


def print_digit(dataset, n):
    data_iter = iter(dataset)
        
    result = data_iter.next()
    images = result['image']
    label = result['label']

    for i in range(n):
        image = np.array(images)[i,:]
        image = image.reshape(28, 28)
        
        plt.figure(n, figsize=(3, 3))
        plt.title(label[i])
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()