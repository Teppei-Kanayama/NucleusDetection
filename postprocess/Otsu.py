# coding: utf-8

import cv2
import matplotlib.pyplot as plt
class OTSU:
    __data = 0
    def __init__(self,file_name):
        src = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        ret, self.__data = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    def show(self):
        plt.imshow(self.__data)
        plt.gray()
        plt.show()
    def data(self):
        return (255. - self.__data)  # 白黒反転

