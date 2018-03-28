
# coding: utf-8

import cv2
import numpy as np
import glob
import random
import shutil
import os
import imagetype_classification as imcl

files = glob.glob("*.png")
type_1 = []
type_2 = []
type_3 = []
type_4 = []
type_5 = []
types = [type_1,type_2,type_3,type_4,type_5]

target_dir_1 = "./train_default"
if not os.path.isdir(target_dir_1):
    os.makedirs(target_dir_1)
target_dir_2 = "./val_default"
if not os.path.isdir(target_dir_2):
    os.makedirs(target_dir_2)

for f in files:
    imagetype = imcl.imagetype_classification(f)
    #それぞれの分類に従ってファイル名のリストを作る
    if imagetype == 1:
        type_1.append(f)
    elif imagetype == 2:
        type_2.append(f)
    elif imagetype == 3:
        type_3.append(f)
    elif imagetype == 4:
        type_4.append(f)
    else:
        type_5.append(f)

for type in types:
    random.shuffle(type)
    #670枚から50枚くらいのvalidationデータを持ってくる
    val_data_size = max(int(len(type)*0.08), 1)
    train_images = type[val_data_size:]
    val_images = type[:val_data_size]
    for image in train_images:
        shutil.copy(image, "./train_default")
    for image in val_images:
        shutil.copy(image, "./val_default")

