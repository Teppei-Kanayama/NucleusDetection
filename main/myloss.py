import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pdb

eps = 0.000001
def weighted_binary_cross_entropy(input, target, weight):
    loss = (target * (input + eps).log() + (1. - target) * (1. - input + eps).log()) * (-1.)
    weighted_loss = weight * loss
    #weight = weight.astype(np.float64) * 9 + 1.0
    return weighted_loss.mean()
