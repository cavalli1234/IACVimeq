import os.path as op
import sys

sys.path.append(op.realpath(op.join(op.split(__file__)[0], "..")))

from data.img_io import load
from keras.datasets import cifar10
from matplotlib.pyplot import imshow, show
from utils.naming import *
from skimage import exposure
from lib.models import *
from lib.training import train_model
import numpy as np
from keras.losses import mean_squared_error as mse
from utils.logging import *


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


def cut_left(data, split):
    return np.array(data[:int(len(data)*split)])


def cut_right(data, split):
    return np.array(data[int(len(data)*split):])


if __name__ == '__main__':
    set_verbosity(DEBUG)
    (train, _), (valid, _) = cifar10.load_data()

    train = train[:40000] / 255.0
    valid = valid[:1000] / 255.0
    image_shape = np.shape(train[0])

    train = (train, make_ground_truth(train))
    valid = (valid, make_ground_truth(valid))

    model = train_model(model_generator=lambda: plain_cnn(layers=5),
                        train=train,
                        valid=valid,
                        loss=mse,
                        patience=5,
                        learning_rate=3e-4,
                        max_epochs=200)

