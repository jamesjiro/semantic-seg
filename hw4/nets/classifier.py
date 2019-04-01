"""
   Here you will implement a relatively shallow neural net classifier on top of
   the hypercolumn (zoomout) features. You can look at a sample MNIST
   classifier here: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from nets.zoomout import *
import numpy as np
from torchvision import transforms

class FCClassifier(nn.Module):
    """
        Fully connected classifier on top of zoomout features.
        Input: extracted zoomout features.
        Output: H x W x 21 softmax probabilities.
    """
    def __init__(self, n_classes=21):
        super(FCClassifier, self).__init__()
        """
        TODO: Implement a fully connected classifier.
        """
        self.layer0 = nn.Linear(1472 * 1, 100)
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(100, 21)
        # You will need to compute these and store as *.npy files
        self.mean = torch.Tensor(np.load("./features/mean.npy"))
        self.std = torch.Tensor(np.load("./features/std.npy"))

    def forward(self, x):
        # normalization
        x = x.type(torch.FloatTensor)
        """
        TODO: Fix standardization
        """
        #x = (x - self.mean)/self.std
        x = self.layer0(x)
        x = self.relu(x)
        x = self.layer1(x)
        return F.log_softmax(x)


class DenseClassifier(nn.Module):
    """
        Convolutional classifier on top of zoomout features.
        Input: extracted zoomout features.
        Output: H x W x 21 softmax probabilities.
    """
    def __init__(self, fc_model, n_classes=21):
        super(DenseClassifier, self).__init__()
        """
        TODO: Convert a fully connected classifier to 1x1 convolutional.
        """
        fcc_modules = fc_model.children()

        # First linear fully connected layer
        fc0 = next(fcc_modules).state_dict()
        self.fc0_in = fc0["weight"].size(1)
        self.fc0_out = fc0["weight"].size(0)
        conv0 = nn.Conv2d(self.fc0_in, self.fc0_out, 1, 1)
        conv0.load_state_dict({"weight":fc0["weight"].view(self.fc0_out, self.fc0_in, 1, 1),
                               "bias":fc0["bias"]})
        self.conv0 = conv0

        # Activation layer
        self.relu = next(fcc_modules)

        # Second linear fully connected layer
        fc1 = next(fcc_modules).state_dict()
        self.fc1_in = fc1["weight"].size(1)
        self.fc1_out = fc1["weight"].size(0)
        conv1 = nn.Conv2d(self.fc1_in, self.fc1_out, 1, 1)
        conv1.load_state_dict({"weight":fc1["weight"].view(self.fc1_out, self.fc1_in, 1, 1),
                               "bias":fc1["bias"]})
        self.conv1 = conv1

        self.mean = np.load("./features/mean.npy")
        self.std = np.load("./features/std.npy")

        # You'll need to add these trailing dimensions so that it broadcasts correctly.
        self.mean = torch.Tensor(np.expand_dims(np.expand_dims(self.mean, -1), -1))
        self.std = torch.Tensor(np.expand_dims(np.expand_dims(self.std, -1), -1))

    def forward(self, x):
        """
        Make sure to upsample back to 224x224 --take a look at F.upsample_bilinear
        """
        #x = F.upsample_bilinear(x, [224, 224])
        #x = x.view(self.fc0_out, self.fc0_in, 224, 224)

        # normalization
        x = (x - self.mean)/self.std
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv1(x)
        return F.log_softmax(x)
