from utils.naming import *
from data.img_io import load
from keras.datasets import cifar10
from skimage import color
import numpy as np


def load_train(num=None, shuffle=True, gray=True):
    (data, _), (_, _) = cifar10.load_data()

    if shuffle:
        np.random.shuffle(data)

    if num is None:
        num = len(data)

    data = data[:num] / 255.0

    return color.rgb2grey(data) if gray else data


def load_valid(num=None, shuffle=True, gray=True):
    (_, _), (data, _) = cifar10.load_data()

    if shuffle:
        np.random.shuffle(data)

    if num is None:
        num = len(data)

    data = data[:num] / 255.0

    return color.rgb2grey(data) if gray else data


def load_test(num=None, shuffle=True, gray=True):
    data = load(dataset_path(), force_format=[240, 220, 1 if gray else 3])
    if shuffle:
        np.random.shuffle(data)

    if num is None:
        num = len(data)

    return data[:num]