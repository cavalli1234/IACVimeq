import os.path as op
import sys

sys.path.append(op.realpath(op.join(op.split(__file__)[0], "..")))

from data.loading import load_train, load_valid
from utils.naming import *
from skimage import exposure, color
from lib.models import *
from lib.training import train_model
import numpy as np
from keras.losses import mean_squared_error as mse
from utils.logging import *
from utils.utility import *


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


def load_data(tn=None, vn=None):
    train = load_train(tn, shuffle=False, gray=True)
    valid = load_valid(vn, shuffle=False, gray=True)
    return train, valid


def preprocess_data_cnn(train, valid):
    train_ = np.expand_dims(train, axis=-1)
    valid_ = np.expand_dims(valid, axis=-1)

    train = (train_, np.expand_dims(make_ground_truth(train), axis=-1))
    valid = (valid_, np.expand_dims(make_ground_truth(valid), axis=-1))

    return train, valid


def preprocess_data_ffnn(train, valid, bins=128, kp=0.5):
    train_ = attach_histogram_to_batch(train, nbins=bins)
    valid_ = attach_histogram_to_batch(valid, nbins=bins)

    train = shuffle_data((train_, make_ground_truth_ffnn(train)), keep_probability=kp)
    valid = shuffle_data((valid_, make_ground_truth_ffnn(valid)), keep_probability=kp)

    return train, valid

if __name__ == '__main__':
    set_verbosity(DEBUG)
    # (train, _), (valid, _) = cifar10.load_data()

    # train = train[:] / 255.0
    # valid = valid[:] / 255.0

    # image_shape = np.shape(train[0])

    # train = np.array([color.rgb2gray(x) for x in train])
    # valid = np.array([color.rgb2gray(x) for x in valid])

    train, valid = load_data(tn=15000, vn=2000)
    train, valid = preprocess_data_ffnn(train, valid, bins=64, kp=0.25)

    # mg = lambda: hist_building_cnn(layers=4, bins=64)
    mg = lambda: ff_hist(65)

    model = train_model(model_generator=mg,
                        train=train,
                        valid=valid,
                        loss=mse,
                        patience=5,
                        learning_rate=1e-4,
                        max_epochs=200,
                        log_images=False)

