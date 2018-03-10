
# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
class ColorPCA:#なお結果画像に変化はなかった模様。
    data = 0
    def __init__(self,org_data):
        org_re = org_data.reshape(-1,4)
        org_mean=np.mean(org_data,axis=(0,1))
        mean, eigenvectors = cv2.PCACompute(
            data=org_re,
            mean=org_mean.reshape(1,-1),
            maxComponents=1
        )
        vec = eigenvectors.reshape(4)
        eigen_data =org_data.dot(vec)
        global data
        data = (eigen_data-np.nanmin(eigen_data))/(np.nanmax(eigen_data)-np.nanmin(eigen_data))*255
    def show(self):
        plt.imshow(data)
        plt.gray()
        plt.show()
        lap= cv2.Laplacian(data,cv2.CV_32F)
        edge_img = cv2.convertScaleAbs(lap)
        plt.imshow(edge_img)
        plt.show()
    def data(self):
        global data
        return data
