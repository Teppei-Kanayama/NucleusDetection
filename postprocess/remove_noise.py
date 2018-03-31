import cv2
import numpy as np
from skimage import morphology
import matplotlib

# raw: 生データ, th: 核として検出する確率値, shape: 元画像の大きさ(height, width), area: 閾値, prob: 閾値
# 面積が11未満の核または、面積がarea未満でかつ核の平均確率がprob未満のものを除去する
def remove_noise(raw, shape, th=0.5, area=30, prob=0.8):
    # binarization
    bin = np.where(raw > th, 1, 0)

    # labeling
    bin_labels = morphology.label(bin, connectivity=1)

    # separate nucleus
    bin_num = np.max(bin_labels)
    bin_individual = np.array([np.where(bin_labels == i+1, 1, 0) for i in range(bin_num)])

    # calculate probs
    probs = np.array([np.sum(raw * bin) / np.sum(bin) for bin in bin_individual])

    # remove noise
    output = np.zeros(shape, dtype = "uint8")
    for i in range(bin_num):
        sum = np.sum(bin_individual[i])
        if sum >= area or (sum >= 11 and probs[i] >= prob):
            output = output + bin_individual[i]

    return output
