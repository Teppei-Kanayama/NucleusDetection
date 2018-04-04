import numpy as np
from PIL import Image
import os
import sys
import cv2
from skimage import morphology

def fill(img):
    label =  morphology.label(img, background=100000000)
    background = np.unique(label, return_counts=True)[1].argmax() + 1
    return (label != background).astype(np.uint8) * 255
