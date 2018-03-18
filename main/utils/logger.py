import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import sys
import os
import pdb
from PIL import Image

class Logger(object):

    def __init__(self, options, iddataset):
        self.options = options
        self.train_loss_list = []
        self.validation_loss_list = []
        self.validation_score_matrix = np.zeros((options.epochs, 10))
        self.validation_score_list = []
        self.iddataset = iddataset
        if not os.path.exists(self.options.save_val + str(self.options.id)):
            os.mkdir(self.options.save_val + str(self.options.id))

    def save_val_filenames(self, filename="valfiles.txt"):
        # validation画像のファイル名を列挙したファイルを生成する。
        val_filenames = ""
        for i in range(len(self.iddataset['val'])):
            val_filenames += self.iddataset['val'][i] + "\n"
        f = open(self.options.save_probs + filename, 'w')
        f.write(val_filenames)
        f.close()

    def save_loss(self, loss, phase="train"):
        # lossの値をリストに保存する。
        if phase == "train":
            self.train_loss_list.append(loss)
        elif phase == "val":
            self.validation_loss_list.append(loss)
        else:
            raise NameError

    def draw_loss_graph(self, save_path):
        # epoch数とlossの関係を表す折れ線グラフを描画する。
        epoch = len(self.train_loss_list)
        epoch_arange = np.arange(1, epoch + 1)
        if not self.options.skip_train:
            plt.plot(epoch_arange, self.train_loss_list, label="train loss")
        plt.plot(epoch_arange, self.validation_loss_list, label="val loss")
        plt.legend() # 凡例を表示
        plt.title("Mean WBCELoss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig(save_path + str(self.options.id) + ".png")
        plt.clf()

    def save_output_mask(self, y_hat, original_size, filename):
        # モデルが出力したマスク画像を保存する。（リサイズする）
        result = Image.fromarray((y_hat * 255).astype(np.uint8))
        result = result.resize(original_size)
        result.save(self.options.save_val + str(self.options.id) + "/" + filename + ".png")

    def save_output_prob(self, prob, original_size, filename):
        # モデルが出力した確率値を出力する。
        np.save(self.options.save_val + str(self.options.id) + "/" + filename + ".npy", prob)
