from skimage.exposure import histogram

import os.path as op
import sys

sys.path.append(op.realpath(op.join(op.split(__file__)[0], "..")))

from source.utils.naming import *
import numpy as np
from skimage import color
from data.img_io import load


def attach_histogram(grayscale_image, nbins: int = 256, normalize: bool = True):
    hist, _ = histogram(grayscale_image, nbins)
    hist = np.array(hist)
    image = np.array(grayscale_image)
    image = image.flatten()
    image = np.reshape(image, newshape=(len(image), 1))
    hist = hist / len(image) if normalize else hist
    hists = [hist for _ in range(len(image))]
    return np.concatenate((image, hists), axis=1)


def attach_histogram_to_batch(batch_rgb, nbins: int = 256, normalize: bool = True):
    data = [attach_histogram(color.rgb2gray(x), nbins, normalize) for x in batch_rgb]
    result = data[0]
    for i in range(len(data) - 1):
        result = np.concatenate((result, data[i + 1]))
    return result


def img_diff(img1, img2, function=np.abs):
    im1 = np.array(img1)
    im2 = np.array(img2)
    result = function(im1 - im2)
    return result


if __name__ == '__main__':
    test = load(dataset_path(), force_format=[240, 220, 3])
    data = attach_histogram_to_batch(test, 128)
    print(np.shape(data))







