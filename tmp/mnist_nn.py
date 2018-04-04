#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
パラメーターを最適化する，MNIST用のニューラルネットワークです．
最適化するパラメーターは
epoch_num: エポック数(1~20)
dropout_r:ドロップアウト率(0~0.9)
h_cnl: 隠れ層のチャンネル数（隠れ層は入力層の何倍のユニットにするか）(0.51~5)
batch_size:バッチサイズ(50,100,200)
conv: 畳み込みにするか(True or False)
layers: 層の数 (1~3)
residual: residual にするか(True or False)
conv_ksize: 畳み込みのフィルターサイズ(5,7,9,11,13)
"""

import numpy as np
import chainer
from chainer import optimizers,datasets
import chainer.functions as F
import chainer.links as L

class mnist_nn(chainer.Chain):
    '''
    mnist分類タスクを行うニューラルネットワーク
    '''
    def __init__(self, dropout_r=0.5, h_cnl=0.51, conv=True, layers=3, residual=True, conv_ksize=9):
        '''
        パラメーターの初期化と
        層の作成（Linkの追加）
        を行います
        '''
        super(mnist_nn, self).__init__()
        self.dropout_r = dropout_r
        self.h_cnl = h_cnl
        self.conv = conv
        self.layers = layers
        self.residual = residual
        self.conv_ksize = conv_ksize

        if conv:
            p_size = int((conv_ksize-1)/2)
            self.add_link('layer{}'.format(0), L.Convolution2D(1,round(h_cnl),(conv_ksize,conv_ksize),pad=(p_size,p_size)))
            for i in range(1,layers):
                self.add_link('layer{}'.format(i), L.Convolution2D(round(h_cnl),round(h_cnl),(conv_ksize,conv_ksize),pad=(p_size,p_size)))
            self.add_link('fin_layer', L.Convolution2D(round(h_cnl),10,(28,28)))
        else:
            self.add_link('layer{}'.format(0), L.Linear(784,round(784*h_cnl)))
            for i in range(1,layers):
                self.add_link('layer{}'.format(i), L.Linear(round(784*h_cnl),round(784*h_cnl)))
            self.add_link('fin_layer', L.Linear(round(784*h_cnl),10))

    def __call__(self, x, train=False):
        '''
        ニューラルネットワーク本体です
        '''
        if self.conv:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, 1, 28, 28)
            h = chainer.Variable(x)
            h = F.dropout(F.relu(self['layer{}'.format(0)](h)),ratio=self.dropout_r)
            for i in range(1,self.layers):
                if self.residual:
                    h = F.dropout(F.relu(self['layer{}'.format(i)](h)),ratio=self.dropout_r) + h
                else:
                    h = F.dropout(F.relu(self['layer{}'.format(i)](h)),ratio=self.dropout_r)
            h = self['fin_layer'](h)[:,:,0,0]
        else:
            h = chainer.Variable(x)
            h = F.dropout(F.relu(self['layer{}'.format(0)](h)),ratio=self.dropout_r)
            for i in range(1,self.layers):
                if self.residual:
                    h = F.dropout(F.relu(self['layer{}'.format(i)](h)),ratio=self.dropout_r) + h
                else:
                    h = F.dropout(F.relu(self['layer{}'.format(i)](h)),ratio=self.dropout_r)
            h = self['fin_layer'](h)
        return h
    def loss(self,x,t):
        '''
        誤差関数は交差エントロピーです
        '''
        x = self.__call__(x, True)
        t = chainer.Variable(t)
        loss = F.softmax_cross_entropy(x,t)
        return loss


def test_mnist_nn(epoch_num=10, dropout_r=0.5, h_cnl=0.51, batch_size=100, conv=True, layers=3, residual=True, conv_ksize=9):
    '''
    パラメーター入れたら，MNISTの誤答率を出す関数
    '''
    #一応シードは固定する（これによって，パラメーターに対して出力はexactに定まる
    np.random.seed(1234)

    #モデル最適化の準備
    model = mnist_nn(dropout_r, h_cnl, conv, layers, residual, conv_ksize)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    #データの用意
    train, test = datasets.get_mnist()
    trainn, testn = len(train), len(test)

    #ログ出力用のlinspace
    logging_num = np.linspace(0,epoch_num*int(trainn/batch_size),11).astype(int)

    iter = 0
    for epoch in range(epoch_num):
        #バッチ作成
        batch_idx = np.array(range(trainn))
        np.random.shuffle(batch_idx)
        for batch_num in np.hsplit(batch_idx,int(trainn/batch_size)):
            batch = train[batch_num]
            x,t = batch[0],batch[1]

            #学習
            model.zerograds()
            loss = model.loss(x,t)
            #念のため
            if np.isnan(loss.data):
                print("pass")
                continue
            loss.backward()
            optimizer.update()

            #ログ出力
            if iter in logging_num:
                print(str(np.where(logging_num==iter)[0][0]*10)+'%','\tcross entropy =',loss.data)
            iter+=1


    #性能評価(メモリ節約のため，ミニバッチで行う)
    batch_idx = np.array(range(testn))
    false_p = []
    for batch_num in np.hsplit(batch_idx,100):
        batch = test[batch_num]
        x,t = batch[0].reshape(len(batch_num), 1, 28, 28),batch[1]
        res = model(x).data.argmax(axis=1)
        false_p.append(np.mean(t!=res))
    false_p = np.mean(false_p)

    print("False {:.2f}%".format(false_p*100))
    return false_p

if __name__ == '__main__':
    test_mnist_nn(epoch_num=10, dropout_r=0, h_cnl=1, batch_size=100, conv=False, layers=2, residual=False, conv_ksize=9)
