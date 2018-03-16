# coding: utf-8
import cv2
import matplotlib.pyplot as plt
class Otsu:
    """
    Args:
        data: numpy 2d array
    """
    __data = 0
    def __init__(self, data):
        ret, self.__data = cv2.threshold(data, 0, 255, cv2.THRESH_OTSU)

    def show(self):
        plt.imshow(self.__data)
        plt.gray()
        plt.show()

    def data(self):
        return (255. - self.__data)  # 白黒反転
