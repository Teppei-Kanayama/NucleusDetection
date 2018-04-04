#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 20:09:10 2017

@author: marshi
"""

import GPyOpt
import numpy as np
import mnist_nn
import os

def f(x):
    '''
    mnist_nn.test_mnist_nnのラッパー関数
    _x[0](1~20) -> epoch_num = int(_x[0])
    _x[1](0~0.9) -> dropout_r = np.float32(_x[1])
    _x[2](0.51~5) -> h_cnl = np.float32(_x[2])
    _x[3](50,100,200) -> batch_size = int(_x[3])
    _x[4](0,1) -> conv = bool(_x[4])
    _x[5](1,2,3) -> layers = int(_x[5])
    _x[6](0,1) -> residual = bool(_x[6])
    _x[7](5,7,9,11,13) -> conv_ksize = int(_x[7])
    '''
    ret = []
    for _x in x:
        print(_x)
        _ret = mnist_nn.test_mnist_nn(epoch_num = int(_x[0]),
                                        dropout_r = np.float32(_x[1]),
                                        #h_cnl = np.float32(_x[2]),
                                        batch_size = int(_x[2]),
                                        conv = bool(_x[3]),
                                        layers = int(_x[4]),
                                        residual = bool(_x[5]),
                                        conv_ksize = int(_x[6]))
        ret.append(_ret)
    ret = np.array(ret)
    return ret

#それぞれの変数の領域を指定
bounds = [{'name': 'epochs', 'type': 'continuous', 'domain': (1,20)},
          {'name': 'dropout_r', 'type': 'continuous', 'domain': (0.0,0.9)},
          #{'name': 'h_cnl', 'type': 'continuous', 'domain': (0.51,5)},
          {'name': 'batch_size', 'type': 'discrete', 'domain': (50,100,200)},
          {'name': 'conv', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'layers', 'type': 'discrete', 'domain': (1,2,3)},
          {'name': 'residual', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'conv_ksize', 'type': 'discrete', 'domain': (5,7,9,11,13)}]

#ベイズ最適化オブジェクト作成
#前にセーブしたX,Yがあるなら途中から行う
#ないなら最初に数サンプルをランダムにサンプリングする
filename = "XY.npz"
if os.path.exists(filename):
    XY = np.load(filename)
    X,Y = XY['x'],XY['y']
    myBopt = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, X=X, Y=Y)
else:
    myBopt = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)

#ベイズ最適化100ステップとステップごとの結果の表示
for i in range(1000):
    print(len(myBopt.X),"Samples")
    myBopt.run_optimization(max_iter=1)
    print(myBopt.fx_opt)
    print(myBopt.x_opt)
    #逐次セーブ
    np.savez(filename, x=myBopt.X, y=myBopt.Y)
