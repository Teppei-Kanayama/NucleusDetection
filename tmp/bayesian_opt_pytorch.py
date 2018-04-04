#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 20:09:10 2017

@author: marshi
"""

import GPyOpt
import numpy as np
import mnist_pytorch
import os

def f(x):

    ret = []
    for _x in x:
        print(_x)
        _ret = mnist_pytorch.train_and_test(lr = float(_x[0]),
                                            momentum = float(_x[1]),
                                            epochs = int(_x[2]),
                                            unit_num = int(_x[3]))
        ret.append(_ret)
    ret = np.array(ret)
    return ret

#それぞれの変数の領域を指定
bounds = [{'name': 'lr', 'type': 'continuous', 'domain': (0.001, 0.5)},
          {'name': 'momentum', 'type': 'continuous', 'domain': (0.0,1.0)},
          {'name': 'epochs', 'type': 'discrete', 'domain': (1, 2, 3, 4, 5)},
          {'name': 'unit_num', 'type': 'discrete', 'domain': (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)}]

#ベイズ最適化オブジェクト作成
#前にセーブしたX,Yがあるなら途中から行う
#ないなら最初に数サンプルをランダムにサンプリングする

myBopt = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)

#ベイズ最適化100ステップとステップごとの結果の表示
for i in range(50):
    print("\n", len(myBopt.X),"Samples")
    myBopt.run_optimization(max_iter=1)
    print("best score: ", myBopt.fx_opt)
    print("best params: ", myBopt.x_opt)
