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
from utils import dense_crf

def predict_img(net, img, fn, gpu=False):

    img = np.array(img)
    img = img[:, :, :3] # alphachannelを削除
    img = utils.normalize(img)
    img = np.transpose(img, axes=[2, 0, 1])
    X = torch.FloatTensor(img).unsqueeze(0)
    X = Variable(X, volatile=True)
    if gpu:
        X = X.cuda()

    # y_hat = net(X)
    # return np.asarray((y_hat > 0.5).data) # y_hatは負値もかなり含んでいる
    y = F.sigmoid(net(X))
    y = F.upsample_bilinear(y, scale_factor=2).data[0][0].cpu().numpy()
    yy = dense_crf(np.array(img).astype(np.uint8), y)
    return yy > 0.5
