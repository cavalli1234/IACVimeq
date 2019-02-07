from skimage.exposure import histogram

import os.path as op
import sys

sys.path.append(op.realpath(op.join(op.split(__file__)[0], "..")))

from utils.naming import *
import numpy as np
from skimage import color
from data.img_io import load


def attach_histogram(grayscale_image, nbins: int = 256, normalize: bool = True):
    hist, _ = histogram(grayscale_image, nbins)
    hist = np.array(hist)
    image = grayscale_image
    image = image.flatten()
    image = np.reshape(image, newshape=(len(image), 1))
    hist = hist / len(image) if normalize else hist
    hists = [hist for _ in range(len(image))]
    return np.concatenate((image, hists), axis=1)


def attach_histogram_to_batch(batch_rgb, nbins: int = 256, normalize: bool = True):
    data = [attach_histogram(color.rgb2gray(x), nbins, normalize) for x in batch_rgb]
    # result = data[0]
    # for i in range(len(data) - 1):
    #     result = np.concatenate((result, data[i + 1]))
    result = np.reshape(np.array(data), (np.size(data)//(nbins+1), nbins+1))
    return result


def shuffle_data(input_data_with_target: tuple, keep_probability: float = 1.0):
    x = input_data_with_target[0]
    t = input_data_with_target[1]
    filtered_x = []
    filtered_t = []
    # can optimize filtering with array indexing
    for i in range(len(x)):
        if np.random.uniform() <= keep_probability:
            filtered_x.append(x[i])
            filtered_t.append(t[i])

    # actual shuffling
    tmp = list(zip(filtered_x, filtered_t))
    np.random.shuffle(tmp)
    filtered_x, filtered_t = zip(*tmp)

    result = (np.array(filtered_x), np.array(filtered_t))
    return result


def img_diff(img1: np.ndarray, img2: np.ndarray, function=np.abs):
    result = function(img1 - img2)
    return result


if __name__ == '__main__':
    test = load(dataset_path(), force_format=[240, 220, 3])
    data = attach_histogram_to_batch(test, 2)
    print(np.shape(data))
    print(data)
    d2 = shuffle_data((data, data), keep_probability=0.1)
    print(np.shape(d2))
    print(d2)







