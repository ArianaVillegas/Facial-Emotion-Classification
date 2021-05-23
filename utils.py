import copy
import os

import numpy as np
from skimage.io import imread, imshow, show
from skimage.transform import resize


def get_dataset(folder):
    X = []
    y = []
    for label in os.listdir(folder):
        path = os.path.join(folder, label)
        for image_name in os.listdir(path):
            image = imread(os.path.join(path, image_name))
            image = resize(image, (150, 150), anti_aliasing=True)
            # imshow(image)
            # show()
            # image = np.sum(np.array(image), axis=2)
            image = np.array(image)
            features = image.flatten()
            X.append(copy.deepcopy(features))
            y.append(label)
    return X, y
