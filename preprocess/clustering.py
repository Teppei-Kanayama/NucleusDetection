
# coding: utf-8

# In[ ]:


import scipy.cluster.hierarchy as hierarchy
import sklearn.cluster as cl
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob


# In[ ]:


GRAY_STD = 0.01 #どこからグレースケール扱いするかの値


# In[ ]:


class Clustering:
    __files = 0
    __means = []
    __stds = []
    __clusters = []
    __n_gray_clusters = 0
    __n_color_clusters = 0
    __data_num = 0
    def __init__(self, fold_name, n_gray_clusters, n_color_clusters):
        self.__n_gray_clusters = n_gray_clusters
        self.__n_color_clusters = n_color_clusters
        self.__files = glob.glob(fold_name)
        self.__data_num = len(self.__files)#画像の個数
        
        gray_index = []
        for i,file in enumerate(self.__files):#何周目か計測
            org = cv2.imread(file)
            redata = np.reshape( org, (-1, 3) )
            mean = np.mean(redata, axis = 0)#画像RGBの平均をとる
            if np.std(mean) <= GRAY_STD:
                gray_index.append(i)
            self.__means.append(mean)
            self.__stds.append( np.std(redata, axis = 0) )
        color_index = [ i for i in range(self.__data_num) if i not in gray_index]
        
        data = np.c_[self.__means, self.__stds]#平均と標準偏差を統合
        
        km_gray = cl.KMeans(n_clusters=n_gray_clusters)#グレースケールのクラスタリング
        graydata = data[gray_index]
        gray_clusters = km_gray.fit(graydata)
        km_color = cl.KMeans(n_clusters=n_color_clusters)#色つきのクラスタリング
        colordata = data[color_index]
        color_clusters = km_color.fit(colordata)
        gray_labels = gray_clusters.labels_
        color_labels = list(map(lambda n:n + n_gray_clusters, color_clusters.labels_))
        
        self.__clusters = [-1] * self.__data_num#self.__clustersを埋める
        for ind,lab in zip(gray_index,gray_labels):
            self.__clusters[ind] = lab
        for ind,lab in zip(color_index,color_labels):
            self.__clusters[ind] = lab
    def labels(self):
        return self.__clusters
    def show(self,number):
        if number < 0 or number >= self.__n_gray_clusters + self.__n_color_clusters:
            print("out of range")
            return
        print("cluster",number)
        clusters = np.array(self.__clusters)
        index = np.where(clusters == number)[0]
        for i in index:
            file = self.__files[i]
            t = plt.imread(file)
            plt.imshow(t)
            plt.show()
    def showAll(self):
        for number in range(self.__n_gray_clusters + self.__n_color_clusters):
            self.show(number)


# In[ ]:


output = Clustering("/Users/mashu/Downloads/stage1_train/*/images/*.png", 2,2)


# In[ ]:


output.show(0)


# In[ ]:


output.show(1)


# In[ ]:


output.show(2)


# In[ ]:


output.show(3)

