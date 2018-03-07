import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn

from utils import load
from utils import utils
from unet import UNet
from torch.autograd import Variable
from torch import optim
from optparse import OptionParser

import numpy as np
import sys
import os
import pdb

def train_net(net, data, save, epochs=5, batch_size=2, lr=0.1, val_percent=0.05,
              cp=True, gpu=False):
    dir_img = data + 'images/'
    dir_mask = data + 'masks/'
    dir_save = save
    pdb.set_trace()
    ids = load.get_ids(dir_img)
    ids = load.split_ids(ids)

    # trainとvalに分ける
    iddataset = utils.split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(cp), str(gpu)))

    N_train = len(iddataset['train'])
    N_val = len(iddataset['val'])

    # 最適化手法を定義
    optimizer = optim.SGD(net.parameters(),
                          lr=lr, momentum=0.9, weight_decay=0.0005)
    # 損失関数を定義
    criterion = nn.BCELoss()

    # 学習開始
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch+1, epochs))
        train = load.get_imgs_and_masks(iddataset['train'], dir_img, dir_mask)
        val = load.get_imgs_and_masks(iddataset['val'], dir_img, dir_mask)
        epoch_loss = 0

        for i, b in enumerate(utils.batch(train, batch_size)):
            X = np.array([i[0] for i in b])[:, :3, :, :] # alpha channelを取り除く
            y = np.array([i[1] for i in b])

            X = torch.FloatTensor(X)
            y = torch.ByteTensor(y)

            if gpu:
                X = X.cuda()
                y = y.cuda()

            X = Variable(X)
            y = Variable(y)

            y_pred = net(X)
            probs = F.sigmoid(y_pred)
            probs_flat = probs.view(-1)
            y_flat = y.view(-1)
            loss = criterion(probs_flat, y_flat.float() / 255.)
            epoch_loss += loss.data[0]

            print('{0:.4f} --- loss: {1:.6f}'.format(i*batch_size/N_train,
                                                     loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss/i))

        if cp:
            torch.save(net.state_dict(),
                       dir_save + 'CP{}.pth'.format(epoch+1))

            print('Checkpoint {} saved !'.format(epoch+1))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-d', '--data', dest='data',
                      default='/data/unagi0/kanayama/dataset/nuclei_images/stage1_train_preprocessed/', help='path to training data')
    parser.add_option('-s', '--save', dest='save',
                      default='/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/',
                      help='path to save models')


    (options, args) = parser.parse_args()

    # モデルの定義
    net = UNet(3, 1)

    # 学習済みモデルをロードする
    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

    if options.gpu:
        net.cuda()

    #学習を実行
    train_net(net, options.data, options.save, options.epochs, options.batchsize, options.lr,
              gpu=options.gpu)
