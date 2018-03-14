# coding: utf-8
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


#白地に黒の画像を想定。黒地に白だったら反転してください
class Devide():#分割アルゴリズム
    __data = 0
    def __init__(self, org_img, org_mask):
        #画像格納
        train_data = org_img
        train_data_gray = cv2.cvtColor(train_data, cv2.COLOR_RGB2GRAY)
        #mask_data = org_mask[:,:,0]
        mask_data = org_mask
        mask_data = mask_data.copy()
        #ヒストグラム正規化により差が出やすくする
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equ = clahe.apply(train_data_gray)
        #equからdatas作成、maskと被ってないとこ消去、ノイズ消去
        datas = []
        for i in range(0,256):
            th = Threshold(equ, i)
            thd = th.data()
            kernel = np.ones((5,5),np.uint8)
            opening = cv2.morphologyEx(thd, cv2.MORPH_OPEN, kernel)#膨張収縮
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)#収縮膨張
            inverse = cv2.bitwise_not(closing)#色調反転
            data = cv2.bitwise_and(inverse, inverse,mask=mask_data)#maskの輪郭外を消去
            datas.append(data)
        datas.append(mask_data)#最後にmaskを入れる

        #輪郭配列を作る。
        all_contours_array = []#dataごとのcontoursが全て入っている
        for data in datas:
            contours_array = []#contour_array:各画像に対して[フラグ,moment,輪郭]の形で複数入っている
            image, contours, hierarchy = cv2.findContours(data,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                curve_flag = flag.no_curve#くびれがなければ0
                if curve(contour):#くびれがあれば1
                    curve_flag = flag.curve
                moment = cv2.moments(contour)
                #重心
                if moment['m00'] != 0:
                    cx = int(moment['m10']/moment['m00'])
                    cy = int(moment['m01']/moment['m00'])
                else:
                    cx = 0
                    cy = 0
                contours_array.append([curve_flag,[cx,cy],contour])
            all_contours_array.append(contours_array)

        seeds = []#watershedアルゴリズムの前景となる輪郭を入れる配列

        first_outer_contours = []
        #datas256番=mask_dataで輪郭を取得。くびれていなければseedsへ。くびれていたらfirst_outer_contoursへ。
        first_contours_array = all_contours_array[256]
        for contour_array in first_contours_array:
            if contour_array[0] == flag.curve:
                first_outer_contours.append(contour_array[2])
            elif contour_array[0] == flag.no_curve:
                seeds.append(contour_array[2])
        #255番からループスタート
        Search(255,first_outer_contours)

        img = np.copy(train_data)
        contour_img = cv2.drawContours(img, seeds, -1, (125,125,0), 1)

        zero = np.zeros(mask_data.shape)
        devided = cv2.fillPoly(zero, pts =seeds, color=(255,255,255))#分割されたやつ
        devided =  np.uint8(devided)
        #前景となる場所を決める
        dist_transform = cv2.distanceTransform(devided,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

        #背景となる場所を決める
        kernel = np.ones((3,3),np.uint8)
        sure_bg = cv2.dilate(mask_data,kernel,iterations=3)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        # 不明領域に0をつけておく
        markers[unknown==255] = 0
        #watershedアルゴリズムを行う
        markers = cv2.watershed(train_data,markers)

        self.__data = markers
    def data(self):
        return self.__data
    def show(self):
        plt.imshow(self.__data)
        plt.show()


class Threshold:#train_data_grayに対して、明度x未満を0、x以上を255に二値化する関数
    __data = 0
    def __init__(self, data, threshold):
        ret, self.__data = cv2.threshold(data,threshold,255,cv2.THRESH_BINARY)
    def data(self):
        return self.__data
    def show(self):
        plt.imshow(self.__data)
        plt.gray()
        plt.show()


DIFF_LIMIT = 0.10
def curve(contour):#くびれがあればTrue,なければFalseを返す関数
    con_t = cv2.contourArea(contour)#その輪郭の面積
    approx = cv2.convexHull(contour)
    con_h = cv2.contourArea(approx)#凸包面積
    if con_h != 0:#0除算を回避
        if con_t/con_h <= 1 - DIFF_LIMIT:
            return True
        else:
            return False


from enum import Enum
class flag(Enum):#フラグ：くびれなし→0,くびれあり→1,検出しない→-1
    no_curve = 0
    curve = 1
    no_detect = -1


def Search(num, outer_contours):#輪郭内のくびれていない輪郭をseedに入れる関数
    contours_array = all_contours_array[num]#data[num]におけるcontours

    for outer_contour in outer_contours:

        #outer_contour内にiのmonentが存在する場合を検出する
        for i in range(len(contours_array)):#輪郭一つ一つに対して
            contour_array = contours_array[i]
            if contour_array[0] != flag.no_detect:#検出対象なら
                if cv2.pointPolygonTest(outer_contour, tuple(contour_array[1]),False) == +1:#輪郭内なら
                    if contour_array[0] == flag.no_curve:#くびれていなければ
                        seeds.append(contour_array[2])#Seedsにその輪郭を追加
                        Delete(num,contour_array[2])#その輪郭内の下部輪郭の削除
    if num > 0:
        return Search(num-1,outer_contours)


def Delete(num,outer_contour): #この階層以下で、outer_contour内にあるやつ全部消去
    for i in range(0, num):#以下の階層において
        contours_array = all_contours_array[i]
        for j in range(len(contours_array)):#輪郭一つ一つに対して
            contour_array = contours_array[j]
            if contour_array[0]!=flag.no_detect:#検出対象なら
                if cv2.pointPolygonTest(outer_contour, tuple(contour_array[1]),False) == +1:#輪郭内なら
                    contour_array[0]=flag.no_detect#検出しないようにする
