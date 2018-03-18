
# coding: utf-8

# In[ ]:



#カラー/モノクロの識別をした後、カラー画像を3つのタイプ（白背景-紫、紫背景-紫、HE染色）に分類するプログラム
#ルールベースでざーっと書いてみました。穴はあると思いますけど、とりあえず。


# In[ ]:

import cv2
import numpy as np
from Otsu import Otsu


# In[ ]:

#カラーかモノクロかをまず判別する
def is_color(b, g, r):
    if np.allclose(b, g) and np.allclose(g,r):
        return False  # monocro
    else:
        return True  # color


# In[ ]:

# モノクロ画像を分類する
def mono_classification(r):
    #R=G=Bなので1つについてのみ画素の平均を計算
    average = r.mean(0).mean(0)

    #全体に明るければ背景は白、暗ければ背景は黒
    if average > 126:
        return 1  # type white-back
    else:
        return 2  # type black-black


# In[ ]:

# カラー画像を分類する
def color_classification(r, g, b, img):
    #RGBそれぞれについて、ヒストグラム（0-255のピクセルがそれぞれいくつあるか）を作成
    #これもnumpy.ndarrayである
    hist_r, bins = np.histogram(r.ravel(),256,[0,256])
    hist_g, bins = np.histogram(g.ravel(),256,[0,256])
    hist_b, bins = np.histogram(b.ravel(),256,[0,256])

    #RGBそれぞれについてヒストグラムのピークを取得
    #r_maxなどはnumpy.int64型であることに注意
    r_max = np.argmax(hist_r)
    b_max = np.argmax(hist_b)
    g_max = np.argmax(hist_g)
    

    if abs(r_max - b_max) < 10 and abs(b_max - g_max) < 10 and abs(g_max - r_max) < 10:
        #3つのピークがほぼ等しい（閾値：10）（＝ほぼ白）なら白背景の紫パターン
        #だけど一応白部分を除いたやつを見て判断
        if maskedhist(img) == 5:
            return 5
        else:
            return 3

    elif b_max - r_max > 7 and r_max - g_max > 7:
        #濃い方からピークがきれいに山3つ、Blue>Red>Greenになるようならば紫背景の紫パターン
        return 4

    elif r_max - b_max > 0 and b_max - g_max > 30:
        # BlueよりRedが濃く、Greenは非常に薄くて山がかなり下なほうなのがHE染色のパターン（肉眼だと赤っぽい）
        #白部分を除いたやつを見て判断
        if maskedhist(img) == 5:
            return 5
        else:
            return 3

    else:
        #白背景の紫パターンで、これで捉えきれないものがあるので（ガバガバふぃるたー）
        if maskedhist(img) == 5:
            return 5
        else:
            return 3


# In[ ]:

#白背景の紫:3とHE：5を識別する
def maskedhist(img):
    img_mono = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #cv2.thresholdが使えるのはグレースケールなので注意
    i = Otsu (img_mono)
    mask = i.data()
    mask_int = mask.astype(np.uint8)
    #cv2.calcHistに与えるmaskはuint8でなければならない
    hist_r_mask = cv2.calcHist([img],[2],mask_int,[256],[0,256])
    hist_g_mask = cv2.calcHist([img],[1],mask_int,[256],[0,256])
    hist_b_mask = cv2.calcHist([img],[0],mask_int,[256],[0,256])
    
    #RGBそれぞれについて（マスクの）ヒストグラムのピークを取得
    r_mask_max = np.argmax(hist_r_mask)
    g_mask_max = np.argmax(hist_g_mask)
    b_mask_max = np.argmax(hist_b_mask)
    
    if r_mask_max - b_mask_max > 7 and b_mask_max - g_mask_max > 10:
        return 5
    #赤が濃ければHE
    else:
        return 3
    


# In[ ]:

# 白黒（背景白）: 1, 白黒（背景黒）:2, 白背景の紫: 3, 紫背景の紫: 4, HE: 5 を返す
def imagetype_classification(filename):
    img = cv2.imread(filename)
    #opencvにおいては画像をimreadするとnumpy.ndarray形式で読み込まれる（行（高さ） x 列（幅） x 色（BGR）の三次元）
    #このとき、RGB順でなくBGR順になることに注意
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    color = is_color(b, g, r)

    if not color:  # モノクロの場合
        return mono_classification(r)
    else: # カラーの場合
        return color_classification(r, g, b, img)


# In[ ]:

if __name__ == "__main__":
    print(imagetype_classification("sample.png"))

