import cv2
import numpy as np
from skimage import morphology
import matplotlib
from PIL import Image

def prob2mask(prob_map, original_image, thresh1=0.5, thresh2=0.6):

    sure_bg = (prob_map > thresh1).astype(np.uint8) * 255
    sure_fg = (prob_map > thresh2).astype(np.uint8) * 255
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown==255] = 0
    #img = np.asarray(original_image.resize((640, 640)))
    img = original_image
    img = img[:, :, :3]
    markers = cv2.watershed(img, markers)
    mask = (markers > 1).astype(np.uint8) * 255

    return mask


# raw: 生データ, th: 核として検出する確率値, shape: 元画像の大きさ(height, width), area: 閾値, prob: 閾値
# 面積が11未満の核または、面積がarea未満でかつ核の平均確率がprob未満のものを除去する
def remove_noise(raw, shape, original_image, th=0.5, area=30, prob=0.8):
    # binarization
    #bin = np.where(raw > th, 1, 0)
    bin = prob2mask(raw, original_image)

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
