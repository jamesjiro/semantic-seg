import sys
import torch
import numpy as np

from torch.utils import data

from nets.zoomout import Zoomout
from data.loader import PascalVOC
from utils import *
import gc

def extract_samples(zoomout, dataset):
    """
    TODO: Follow the directions in the README
    to extract a dataset of 1x1xZ features along with their labels.
    Predict from zoomout using:
         with torch.no_grad():
            zoom_feats = zoomout(images.cpu().float().unsqueeze(0))
    """
    features = []
    features_labels = []

    for image_idx in range(len(dataset)):
        print("...Processing image %d / %d" % (image_idx, len(dataset)), end='\r')
        images, labels = dataset[image_idx]
        with torch.no_grad():
            zoom_feats = zoomout(images.cpu().float().unsqueeze(0))
            for label in range(21):
                if (labels == label).any():
                    idcs = (labels == label).nonzero()
                    n_idcs = idcs.size()[0]
                    n_samples = min(3, n_idcs)
                    sample_idcs = np.random.randint(0, n_idcs - 1, n_samples)
                    for idx in sample_idcs:
                        feat = zoom_feats[0, :, idcs[idx][0], idcs[idx][1]]
                        features.append(feat.numpy().tolist())
                        features_labels.append(label)

    features = np.asarray(features)
    features_labels = np.asarray(features_labels)
    return features, features_labels


def main():
    zoomout = Zoomout().cpu().float()
    for param in zoomout.parameters():
        param.requires_grad = False

    dataset_train = PascalVOC(split = 'train')

    features, labels = extract_samples(zoomout, dataset_train)

    np.save("./features/feats_x.npy", features)
    np.save("./features/feats_y.npy", labels)
    np.save(".features/mean.npy", np.mean(features, axis=0))
    np.save(".features/std.npy", np.std(features, axis=0))


if __name__ == '__main__':
    main()
