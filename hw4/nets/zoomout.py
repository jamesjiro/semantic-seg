"""
TODO: Implement zoomout feature extractor.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models

class Zoomout(nn.Module):
    def __init__(self):
        super(Zoomout, self).__init__()

        # load the pre-trained ImageNet CNN and list out the layers
        self.vgg = models.vgg11(pretrained=True)
        self.feature_list = list(self.vgg.features.children())

        """
        TODO:  load the correct layers to extract zoomout features.
        """
        self.layer0 = nn.Sequential(*self.feature_list[0:2])
        self.layer1 = nn.Sequential(*self.feature_list[2:5])
        self.layer2 = nn.Sequential(*self.feature_list[5:10])
        self.layer3 = nn.Sequential(*self.feature_list[10:15])
        self.layer4 = nn.Sequential(*self.feature_list[15:20])
    def forward(self, x):
        out0 = self.layer0(x)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        """
        TODO: load the correct layers to extract zoomout features.
        Hint: use F.upsample_bilinear and then torch.cat.
        """
        out0 = F.upsample_bilinear(out0, [112,112])
        out1 = F.upsample_bilinear(out1, [112,112])
        out2 = F.upsample_bilinear(out2, [112,112])
        out3 = F.upsample_bilinear(out3, [112,112])
        out4 = F.upsample_bilinear(out4, [112,112])
        desc = torch.cat([out0, out1, out2, out3, out4], 1)

        return desc
