import os.path as op
import sys

sys.path.append(op.realpath(op.join(op.split(__file__)[0], "..")))

import keras.models as km
from keras.datasets import cifar10
from lib.models import *
from utils.naming import *
import numpy as np
from matplotlib.pyplot import imshow, show
from skimage import exposure, color
from data.img_io import load
from utils.utility import img_diff

def showimgs(imgs):
    for img in imgs:
        imshow(img)
        show()


def make_ground_truth(imgs: np.ndarray):
    out = np.zeros_like(imgs)
    for idx in range(len(imgs)):
        tmp = exposure.equalize_hist(imgs[idx])
        # returned equalized image is in 0-1 floating points!
        if out.dtype == np.uint8:
            tmp *= 255
        out[idx] = tmp
    return out


if __name__ == '__main__':
    # model = km.load_model(models_path("plain_cnn_L5.h5"))
    model = plain_cnn(layers=5)
    model.load_weights(models_path("plain_cnn_L5.h5"))

    # (_, _), (test, _) = cifar10.load_data()
    # np.random.shuffle(test)
    # test = test[:3] / 255.0

    test = load(dataset_path(), force_format=[240, 220, 3])

    testgt = make_ground_truth(test)
    testpred = model.predict(test)

    mixed = np.concatenate((test, testpred, testgt), axis=2)
    showimgs(mixed)
