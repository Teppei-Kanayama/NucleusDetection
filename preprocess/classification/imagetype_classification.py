
# coding: utf-8

# In[ ]:



#カラー/モノクロの識別をした後、カラー画像を3つのタイプ（白背景-紫、紫背景-紫、HE染色）に分類するプログラム
#ルールベースでざーっと書いてみました。穴はあると思いますけど、とりあえず。

import cv2
import numpy as np
#from matplotlib import pyplot as plt


# In[ ]:

img = cv2.imread("sample7.png")
#opencvにおいては画像をimreadするとnumpy.ndarray形式で読み込まれる（行（高さ） x 列（幅） x 色（BGR）の三次元）
#このとき、RGB順でなくBGR順になることに注意


# In[ ]:

b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]


# In[ ]:

#カラーかモノクロかをまず判別する
def colorormono(b, g, r):
    if np.allclose(b, g) and np.allclose(g,r):
        return "mono"
    else:
        return "color"


# In[ ]:

corm_result = colorormono(b, g, r)
#print(corm_result)


# In[ ]:

def mono_classification(r):
    average = r.mean(0).mean(0)
    #R=G=Bなので1つについてのみ画素の平均を計算
    
    #jupyter内で元画像を表示するためのやつ
    from IPython.display import Image, display_png
    display_png(Image("sample7.png"))
    
    #全体に明るければ背景は白、暗ければ背景は黒
    if average > 126:
        print("This image is type white-back.")
    else:
        print("This image is type black-back.")


# In[ ]:

def color_classification(r, g, b):
    hist_r, bins = np.histogram(r.ravel(),256,[0,256])
    hist_g, bins = np.histogram(g.ravel(),256,[0,256])
    hist_b, bins = np.histogram(b.ravel(),256,[0,256])
    #RGBそれぞれについて、ヒストグラム（0-255のピクセルがそれぞれいくつあるか）を作成
    #これもnumpy.ndarrayである

    #RGBそれぞれについてヒストグラムのピークを取得
    #r_maxなどはnumpy.int64型であることに注意
    r_max = np.argmax(hist_r)
    b_max = np.argmax(hist_b)
    g_max = np.argmax(hist_g)
    
    #jupyter内で元画像を表示するためのやつ
    from IPython.display import Image, display_png
    display_png(Image("sample7.png"))

    if abs(r_max - b_max) < 10 and abs(b_max - g_max) < 10 and abs(g_max - r_max) < 10:
        print("This picture is type white-purple.")
    #3つのピークがほぼ等しい（閾値：10）（＝ほぼ白）なら白背景の紫パターン
    elif b_max - r_max > 7 and r_max - g_max > 7:
        print("This picture is type purple-purple.")
    #濃い方からピークがきれいに山3つ、Blue>Red>Greenになるようならば紫背景の紫パターン
    elif r_max - b_max > 0 and b_max - g_max > 30:
        print("This picture is type HE.")
    #BlueよりRedが濃く、Greenは非常に薄くて山がかなり下なほうなのがHE染色のパターン（肉眼だと赤っぽい）
    else:
        print("This picture is type white-purple(else).")
    #白背景の紫パターンで、これで捉えきれないものがあるので（ガバガバふぃるたー）


# In[ ]:

if corm_result == "mono":
    mono_classification(r)
elif corm_result == "color":
    color_classification(r, g, b)
else:
    print("error!")

