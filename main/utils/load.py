
#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
import numpy as np

from PIL import Image
from functools import partial
from .utils import normalize
import pdb


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return [f.split(".")[0] for f in os.listdir(dir)]


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return [(id, i) for i in range(n) for id in ids]


def to_cropped_imgs(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = Image.open(dir + id + suffix)
        yield np.asarray(im)


def get_imgs_and_masks(ids, dir_img, dir_mask):
    """Return all the couples (img, mask)"""

    pdb.set_trace()
    imgs = to_cropped_imgs(ids, dir_img, '.png')

    # need to transform from HWC to CHW
    imgs_switched = map(partial(np.transpose, axes=[2, 0, 1]), imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '.png')

    return zip(imgs_normalized, masks)
