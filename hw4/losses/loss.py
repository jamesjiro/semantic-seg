import torch
import torch.nn.functional as F

def cross_entropy2d(predicts, targets):
    """
    Hint: look at torch.nn.NLLLoss.  Considering weighting
    classes for better accuracy.
    """
    loss = F.nll_loss(predicts, targets)
    return loss

def cross_entropy1d(predicts, targets):
    loss = F.nll_loss(predicts, targets)
    return loss
