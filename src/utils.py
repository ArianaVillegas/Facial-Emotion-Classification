import copy
import os

import numpy as np
import pywt
from skimage.io import imread


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_dataset(folder):
    X = []
    y = []
    for emotion in os.listdir(folder):
        path = os.path.join(folder, emotion)
        for image_name in os.listdir(path):
            image = imread(os.path.join(path, image_name), as_gray=True)
            coeff = pywt.dwt2(image, 'haar')
            features = np.ndarray.flatten(coeff[0])
            X.append(copy.deepcopy(features))
            y.append(emotion)
    return X, y


def Bootstrap_split(X, n_splits=10, train_size=0.7):
    bt_train = []
    bt_test = []
    for i in range(n_splits):
        train = np.random.choice(len(X), int(len(X)*train_size))
        test = [i for i in range(len(X)) if i not in train]
        bt_train.append(train)
        bt_test.append(test)
    return bt_train, bt_test

