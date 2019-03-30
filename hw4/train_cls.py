import sys
import torch
import argparse
import numpy as np
from PIL import Image
import json
import random
from scipy.misc import toimage, imsave

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

from losses.loss import *
from nets.classifier import FCClassifier

from data.loader import PascalVOC
import torch.optim as optim
from utils import *

def train(dataset, model, optimizer, epoch):
    """
    TODO: Implement training for simple FC classifier.
        Input: Z-dimensional vector
        Output: label.
    """

    batch_size = 64# Can be fairly large

    data_x, data_y = dataset

    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)

    trainset = TensorDataset(data_x, data_y)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=2)

    model.train()

    """
    Put train loop here.
    """
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = cross_entropy1d(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Finished Training')

    torch.save(model, "./models/fc_cls.pkl")


def main():

    classifier = FCClassifier().float()

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)# pick an optimizer.

    dataset_x = np.load("./features/feats_x.npy")
    dataset_y = np.load("./features/feats_y.npy")

    num_epochs = 20# your choice, try > 10

    for epoch in range(num_epochs):
        train([dataset_x, dataset_y], classifier, optimizer, epoch)

if __name__ == '__main__':
    main()
