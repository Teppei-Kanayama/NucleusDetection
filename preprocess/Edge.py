import cv2
import matplotlib.pyplot as plt
#edge検出を行うクラス
class edgeDetect:
    __edge_img = 0
    def __init__(self,org_img):#元画像はカラー
        gray_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2GRAY)
        lap= cv2.Laplacian(gray_img,cv2.CV_64F)
        self.__edge_img = cv2.convertScaleAbs(lap)
    def show(self):#表示
        plt.imshow(self.__edge_img)
        plt.gray()
        plt.show()
    def data(self):#変換
        return self.__edge_img
