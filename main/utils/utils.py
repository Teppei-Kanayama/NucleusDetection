import PIL
import numpy as np
import random
import matplotlib.pyplot as plt
import pdb

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i+1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.seed(42)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255


def data_augmentation(X, y, w):
    k = np.random.randint(4)
    is_flip = np.random.randint(2)
    X = np.rot90(X, k=k, axes=(2, 3)).copy()
    y = np.rot90(y, k=k, axes=(1, 2)).copy()
    w = np.rot90(y, k=k, axes=(1, 2)).copy()
    if is_flip:
        X = np.flip(X, axis=2).copy()
        y = np.flip(y, axis=1).copy()
        w = np.flip(w, axis=1).copy()
    return X, y, w
