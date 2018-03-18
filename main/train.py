import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn

from utils import load, utils, option_manager
from utils.logger import Logger
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


def train_net(options):
    dir_img = options.data + '/images/'
    dir_mask = options.data + '/masks/'
    dir_edge = options.data + '/edges/'
    dir_save_model = options.save_model
    dir_save_state = options.save_state
    ids = load.get_ids(dir_img)

    # trainとvalに分ける  # ここで順序も決まってしまう
    iddataset = utils.split_train_val(ids, options.val_percent)
    N_train = len(iddataset['train'])
    N_val = len(iddataset['val'])
    N_batch_per_epoch_train = int(N_train / options.batchsize)
    N_batch_per_epoch_val = int(N_val / options.val_batchsize)

    # 実験条件の表示
    option_manager.display_info(options, N_train, N_val)

    # 結果の記録用インスタンス
    logger = Logger(options, iddataset)

    #if options.save_probs is not None:
    #    logger.save_val_filenames(iddataset)

    # モデルの定義
    net = UNet(3, 1)

    # 最適化手法を定義
    optimizer = optim.Adam(net.parameters())

    # 学習済みモデルをロードする
    if options.load_model:
        net.load_state_dict(torch.load(options.load_model))
        optimizer.load_state_dict(torch.load(options.load_state))
        print('Model loaded from {}'.format(options.load_model))
        print('State loaded from {}'.format(options.load_state))

    if options.gpu:
        net.cuda()

    # 学習開始
    for epoch in range(options.epochs):
        print('Starting epoch {}/{}.'.format(epoch+1, options.epochs))
        train = load.get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, dir_edge, options.resize_shape)
        original_sizes = load.get_original_sizes(iddataset['val'], dir_img, '.png')
        val = load.get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, dir_edge, options.resize_shape, train=False)
        train_loss = 0
        validation_loss = 0
        validation_score = 0
        validation_scores = np.zeros(10)

        # training phase
        if not options.skip_train:
            net.train()
            for i, b in enumerate(utils.batch(train, options.batchsize)):
                X = np.array([j[0] for j in b])
                y = np.array([j[1] for j in b])
                w = np.array([j[2] for j in b])

                if X.shape[0] != options.batchsize:  # batch sizeを揃える（揃ってないとなぜかエラーになる）
                    continue

                X, y, w = utils.data_augmentation(X, y, w)

                X = torch.FloatTensor(X)
                y = torch.ByteTensor(y)
                w = torch.ByteTensor(w)

                if options.gpu:
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
                loss = weighted_binary_cross_entropy(probs_flat, y_flat.float() / 255., weight)
                train_loss += loss.data[0]

                print('{0:.4f} --- loss: {1:.6f}'.format(i*options.batchsize/N_train,
                                                         loss.data[0]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                break

            print('Epoch finished ! Loss: {}'.format(train_loss/N_batch_per_epoch_train))
            logger.save_loss(train_loss/N_batch_per_epoch_train, phase="train")

        # validation phase
        net.eval()
        probs_array = np.zeros((N_val, options.resize_shape[0], options.resize_shape[1]))
        for i, b in enumerate(utils.batch(val, options.val_batchsize)):
            X = np.array([j[0] for j in b])[:, :3, :, :] # alpha channelを取り除く
            y = np.array([j[1] for j in b])
            w = np.array([j[2] for j in b])
            X = torch.FloatTensor(X)
            y = torch.ByteTensor(y)
            w = torch.ByteTensor(w)

            if options.gpu:
                X = X.cuda()
                y = y.cuda()
                w = w.cuda()

            X = Variable(X, volatile=True)
            y = Variable(y, volatile=True)
            w = Variable(w, volatile=True)

            y_pred = net(X)
            probs = F.sigmoid(y_pred)

            # 予測値を記録しておく
            #probs_array[i] = np.array(probs.data[0][0])

            probs_flat = probs.view(-1)
            y_flat = y.view(-1)
            w_flat = w.view(-1)

            # edgeに対して重み付けをする
            weight = (w_flat.float() / 255.) * 4. + 1.
            loss = weighted_binary_cross_entropy(probs_flat, y_flat.float() / 255., weight)
            validation_loss += loss.data[0]

            y_hat = np.asarray((probs > 0.5).data)
            y_hat = y_hat.reshape((y_hat.shape[2], y_hat.shape[3]))
            y_truth = np.asarray(y.data)

            """
            # calculate validatation score
            if options.calc_score and False:  # 初期はノイズが多くスコアリングに時間がかかるため
                print("Image No.", i, "started.")
                score, scores, _ = validate(y_hat, y_truth)
                validation_score += score
                validation_scores += scores
            """

            logger.save_output_mask(y_hat, original_sizes[i], iddataset['val'][i])
            if options.save_probs is not None:
                logger.save_output_prob(np.asarray(probs.data[0][0]), original_sizes[i], iddataset['val'][i])

            break

        print('Val Loss: {}'.format(validation_loss / N_batch_per_epoch_val))
        logger.save_loss(validation_loss / N_batch_per_epoch_val, phase="val")

        """
        if options.calc_score:
            print('score: {}'.format(validation_score / i))
            validation_score_matrix[epoch] = validation_scores / i
            validation_score_list.append(validation_score / i)
        else:
            validation_score_list.append(0.0)
        """

        # modelとoptimizerの状態を保存する。
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(),
                       dir_save_model + str(options.id) + '_CP{}.model'.format(epoch+1))
            torch.save(optimizer.state_dict(),
                       dir_save_state + str(options.id) + '_CP{}.state'.format(epoch+1))

            print('Checkpoint {} saved !'.format(epoch+1))

        logger.draw_loss_graph("./results/loss")

        """
        # draw score graph
        if options.calc_score:
            plt.plot(epoch_arange, validation_score_list, label="Mean", linewidth=4)
            threshold = np.linspace(0.5, 0.95, 10)
            threshold = np.around(threshold, decimals=2)
            for i, thresh in enumerate(threshold):
                plt.plot(epoch_arange, validation_score_matrix[:epoch+1, i], label=str(thresh))

            plt.legend()  # 凡例を表示
            plt.title("Average Precision")
            plt.xlabel("epochs")
            plt.ylabel("score")
            plt.savefig("./results/score.png")
            plt.clf()
        """

if __name__ == '__main__':
    (options, args) = option_manager.parse()

    # 学習を実行
    train_net(options)
