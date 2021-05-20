import copy
import os

import numpy as np
from skimage.io import imread


def get_dataset(folder):
    X = []
    y = []
    for emotion in os.listdir(folder):
        path = os.path.join(folder, emotion)
        for image_name in os.listdir(path):
            image = imread(os.path.join(path, image_name), as_gray=True)
            features = np.reshape(image, (48 * 48))
            X.append(copy.deepcopy(features))
            y.append(emotion)
    return X, y
