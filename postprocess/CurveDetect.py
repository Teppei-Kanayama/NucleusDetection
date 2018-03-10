# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
DIFF_LIMIT = 0.15  # 輪郭面積と凸包面積の差の限界値。要調整
class CurveDetect:
    __data = 0
    __contours = 0
    __concaves = []
    __rects = []
    def __init__(self, data):
        #mask = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)#RGBで入ってきた時の保険。グレースケールで入ってくることが保証できるなら不要
        mask = data
        self.__data = data#元のマスクを格納する
        image, self.__contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in self.__contours:
            con_t = cv2.contourArea(contour)#その輪郭の面積
            approx = cv2.convexHull(contour)
            con_h = cv2.contourArea(approx)#凸包面積
            if con_h != 0:#0除算を回避
                if con_t/con_h <= 1 - DIFF_LIMIT:
                    self.__concaves.append(contour)
            self.__rects.append(approx)
    def __drawdata(self,contours):
        img = np.copy(self.__data)
        contour_img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
        plt.imshow(contour_img)
        plt.show()
    def show(self):
        self.__drawdata(self.__contours)
        self.__drawdata(self.__rects)
        self.__drawdata(self.__concaves)
    def data(self):
        return self.__concaves

if __name__ == "__main__":
    d = cv2.imread("test.png")
    cu = CurveDetect(d)
    cu.show()
