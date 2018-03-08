import numpy as np
from PIL import Image
import os
import sys
import cv2

def morphology(predicted_mask_array, iterations=1):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(predicted_mask_array, kernel, iterations=iterations)
    dilation = cv2.dilate(erosion, kernel, iterations=iterations)
    return dilation
