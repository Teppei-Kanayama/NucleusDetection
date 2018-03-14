import numpy as np
from PIL import Image
import cv2

# 膨張圧縮処理で小さいノイズを消す
def delete_noise(predicted_mask_array, iterations):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(predicted_mask_array, kernel, iterations=iterations)
    dilation = cv2.dilate(erosion, kernel, iterations=iterations)
    return dilation


# validation画像の場合は、元画像に予測値と正解を重ねて表示する
# test画像の場合は、元画像に予測値のみ表示する
def show_prediction(original_image_array, predicted_mask_array, gt_mask_array=None, iterations=0):

    _, contours, hierarchy = cv2.findContours(predicted_mask_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dst = cv2.drawContours(original_image_array, contours, -1, (0, 255, 0), 1) #GREEN

    if gt_mask_array is not None:
        _, contours, hierarchy = cv2.findContours(gt_mask_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        dst = cv2.drawContours(dst, contours, -1, (255, 0, 0), 1) #RED

    return Image.fromarray(dst)
