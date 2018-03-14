import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy
from PIL import Image
import argparse
import os
import pdb
import numpy as np

import sys
sys.path.append("../main/")
from utils import utils

def predict_img(net, img, gpu):

    img = img[:, :, :3] # alphachannelを削除
    img = utils.normalize(img)
    img = np.transpose(img, axes=[2, 0, 1])
    X = torch.FloatTensor(img).unsqueeze(0)
    X = Variable(X, volatile=True)
    if gpu:
        X = X.cuda()

    y_hat = F.sigmoid(net(X))
    y_hat = np.asarray((y_hat > 0.5).data)
    y_hat = y_hat.reshape((y_hat.shape[2], y_hat.shape[3]))
    return y_hat
