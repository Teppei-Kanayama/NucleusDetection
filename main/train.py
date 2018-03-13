import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
from PIL import Image

from myloss import weighted_binary_cross_entropy
from validation import validate

SIZE = (640, 640)

def train_net(net, data, save, save_val, epochs=5, batch_size=2, val_batch_size=1, lr=0.1, val_percent=0.05, cp=True, gpu=False, calc_score=True):
    dir_img = data + '/images/'
    #dir_img = data + '_gray/images/'
    #dir_img = data + '_color/images/'
    dir_mask = data + '/masks/'
    dir_edge = data + '/edges/'
    dir_save = save
    ids = load.get_ids(dir_img)
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
    N_batch_per_epoch_train = int(N_train / batch_size)
    N_batch_per_epoch_val = int(N_val / val_batch_size)
    # 最適化手法を定義
    #optimizer = optim.SGD(net.parameters(),
    #                      lr=lr, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters())
    criterion = nn.BCELoss()

    # 学習開始
    train_loss_list = []
    validation_loss_list = []
    validation_score_matrix = np.zeros((epochs, 10))
    validation_score_list = []


    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch+1, epochs))
        train = load.get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, dir_edge, SIZE)
        original_sizes = load.get_original_sizes(iddataset['val'], dir_img, '.png')
        val = load.get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, dir_edge, SIZE)
        train_loss = 0
        validation_loss = 0
        validation_score = 0
        validation_scores = np.zeros(10)

        # training phase
        net.train()
        for i, b in enumerate(utils.batch(train, batch_size)):
            X = np.array([j[0] for j in b])
            y = np.array([j[1] for j in b])
            w = np.array([j[2] for j in b])

            X, y, w = utils.data_augmentation(X, y, w)

            X = torch.FloatTensor(X)
            y = torch.ByteTensor(y)
            w = torch.ByteTensor(w)

            if gpu:
                X = X.cuda()
                y = y.cuda()
                w = w.cuda()

            X = Variable(X)
            y = Variable(y)
            w = Variable(w)

            y_pred = net(X)
            probs = F.sigmoid(y_pred)
            probs_flat = probs.view(-1)
            y_flat = y.view(-1)
            w_flat = w.view(-1)
            weight = (w_flat.float() / 255.) * 4. + 1.
            #weight = (w_flat.float() / 255.) * 0. + 1.
            #loss = weighted_binary_cross_entropy(probs_flat, y_flat.float() / 255., weight)
            loss = criterion(probs_flat, y_flat.float() / 255.)
            #print(np.isclose(loss.data[0], loss2.data[0]))
            train_loss += loss.data[0]

            print('{0:.4f} --- loss: {1:.6f}'.format(i*batch_size/N_train,
                                                     loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(train_loss/N_batch_per_epoch_train))
        train_loss_list.append(train_loss/N_batch_per_epoch_train)

        # validation phase
        net.eval()
        for i, b in enumerate(utils.batch(val, val_batch_size)):
            X = np.array([j[0] for j in b])[:, :3, :, :] # alpha channelを取り除く
            y = np.array([j[1] for j in b])
            w = np.array([j[2] for j in b])
            X = torch.FloatTensor(X)
            y = torch.ByteTensor(y)
            w = torch.ByteTensor(w)

            if gpu:
                X = X.cuda()
                y = y.cuda()
                w = w.cuda()

            X = Variable(X, volatile=True)
            y = Variable(y, volatile=True)
            w = Variable(w, volatile=True)

            y_pred = net(X)
            probs = F.sigmoid(y_pred)
            probs_flat = probs.view(-1)
            y_flat = y.view(-1)
            w_flat = w.view(-1)

            # edgeに対して重み付けをする
            weight = (w_flat.float() / 255.) * 4. + 1.
            #weight = (w_flat.float() / 255.) * 0. + 1.
            #loss = weighted_binary_cross_entropy(probs_flat, y_flat.float() / 255., weight)
            loss = criterion(probs_flat, y_flat.float() / 255.)
            #print(np.isclose(loss.data[0], loss2.data[0]))
            validation_loss += loss.data[0]

            y_hat = np.asarray((probs > 0.5).data)
            y_hat = y_hat.reshape((y_hat.shape[2], y_hat.shape[3]))
            y_truth = np.asarray(y.data)

            # calculate validatation score
            if calc_score and False:  # 初期はノイズが多くスコアリングに時間がかかるため
                print("Image No.", i, "started.")
                score, scores, _ = validate(y_hat, y_truth)
                validation_score += score
                validation_scores += scores

            # save images
            result = Image.fromarray((y_hat * 255).astype(np.uint8))
            result = result.resize(original_sizes[i])
            result.save(save_val + iddataset['val'][i] + ".png")

        print('Val Loss: {}'.format(validation_loss / N_batch_per_epoch_val))
        validation_loss_list.append(validation_loss / N_batch_per_epoch_val)

        if calc_score:
            print('score: {}'.format(validation_score / i))
            validation_score_matrix[epoch] = validation_scores / i
            validation_score_list.append(validation_score / i)
        else:
            validation_score_list.append(0.0)

        if cp and (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(),
                       dir_save + '4_CP{}.pth'.format(epoch+1))

            print('Checkpoint {} saved !'.format(epoch+1))

        epoch_arange = np.arange(1, epoch + 2)
        # draw loss graph
        plt.plot(epoch_arange, train_loss_list, label="train loss")
        plt.plot(epoch_arange, validation_loss_list, label="val loss")
        plt.legend() # 凡例を表示
        plt.title("Mean WBCELoss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig("./results/loss4.png")
        plt.clf()

        # draw score graph
        if calc_score:
            plt.plot(epoch_arange, validation_score_list, label="Mean", linewidth=4)
            threshold = np.linspace(0.5, 0.95, 10)
            threshold = np.around(threshold, decimals=2)
            for i, thresh in enumerate(threshold):
                plt.plot(epoch_arange, validation_score_matrix[:epoch+1, i], label=str(thresh))

            plt.legend() # 凡例を表示
            plt.title("Average Precision")
            plt.xlabel("epochs")
            plt.ylabel("score")
            plt.savefig("./results/score.png")
            plt.clf()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batchsize', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-v', '--val_batch_size', dest='val_batch_size', default=1,
                      type='int', help='validation batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-d', '--data', dest='data',
                      default='/data/unagi0/kanayama/dataset/nuclei_images/stage1_train_default', help='path to training data')
    parser.add_option('-s', '--save', dest='save',
                      default='/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/',
                      help='path to save models')
    parser.add_option('--save_val', dest='save_val',
                      default='/data/unagi0/kanayama/dataset/nuclei_images/answer_val/',
                      help='path to save models')
    parser.add_option('--calc_score', dest='calc_score',
                      default=True,
                      help='whether calculate score or not')

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
    train_net(net, options.data, options.save, options.save_val, options.epochs, options.batchsize, options.val_batch_size, options.lr,
              gpu=options.gpu, calc_score=options.calc_score)
