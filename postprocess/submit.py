import numpy as np
from PIL import Image
import os
import pandas as pd
import pdb
from skimage.morphology import label

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

ANSWER_PATH = '/data/unagi0/kanayama/dataset/nuclei_images/answer/'

new_test_ids = []
rles = []

for filename in os.listdir(ANSWER_PATH):
    im = Image.open(ANSWER_PATH + filename)
    im = np.asarray(im).astype(np.float32) / 255.
    rle = list(prob_to_rles(im))
    rles.extend(rle)
    new_test_ids.extend([filename.split(".")[0]] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('submit.csv', index=False)
