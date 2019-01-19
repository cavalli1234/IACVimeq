import os.path as op
import sys

sys.path.append(op.realpath(op.join(op.split(__file__)[0], "..")))

from data.img_io import load
from keras.datasets import cifar10
from matplotlib.pyplot import imshow, show
from utils.naming import *
from skimage import exposure, color
from lib.models import *
from lib.training import train_model
import numpy as np
from keras.losses import mean_squared_error as mse
from utils.logging import *
from utils.utility import *


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


def make_ground_truth_ffnn(rgb_imgs: np.ndarray):
    gt = make_ground_truth(np.array(rgb_imgs))
    return gt.flatten()


def cut_left(data, split):
    return np.array(data[:int(len(data)*split)])


def cut_right(data, split):
    return np.array(data[int(len(data)*split):])


if __name__ == '__main__':
    set_verbosity(DEBUG)
    (train, _), (valid, _) = cifar10.load_data()

    train = train[:7000] / 255.0
    valid = valid[:2000] / 255.0

    image_shape = np.shape(train[0])

    train = np.array([color.rgb2gray(x) for x in train])
    valid = np.array([color.rgb2gray(x) for x in valid])

    print(np.shape(train))

    train_ = attach_histogram_to_batch(train, nbins=128)
    valid_ = attach_histogram_to_batch(valid, nbins=128)

    # train_ = np.expand_dims(train, axis=-1)
    # valid_ = np.expand_dims(valid, axis=-1)

    train = (train_, make_ground_truth_ffnn(train))
    valid = (valid_, make_ground_truth_ffnn(valid))
    # train = (train_, np.expand_dims(make_ground_truth(train), axis=-1))
    # valid = (valid_, np.expand_dims(make_ground_truth(valid), axis=-1))
    print(np.shape(train[0]), np.shape(train[1]))

    model = train_model(model_generator=lambda: ff_hist(n_inputs=129, name='ddddummy2'),
                        train=train,
                        valid=valid,
                        loss=mse,
                        patience=5,
                        learning_rate=1e-4,
                        max_epochs=200,
                        log_images=False)

