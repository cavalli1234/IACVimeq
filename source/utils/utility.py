from skimage.exposure import histogram

import os.path as op
import sys

sys.path.append(op.realpath(op.join(op.split(__file__)[0], "..")))

from utils.naming import *
import numpy as np
from skimage import color
from data.img_io import load
from lib.model_wrap.model_wrapper import ModelWrapper
import tqdm
from utils.logging import log
from data.loading import *
from skimage import exposure, color


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


def build_losses_array(mw: ModelWrapper, data: np.ndarray):
    datalen = len(data)
    out = np.zeros(shape=(datalen,))
    batch_size = 100
    log("Evaluating losses statistics on data...")
    for idx in tqdm.trange(int(np.ceil(datalen/batch_size))):
        low = idx * batch_size
        high = min((idx + 1) * batch_size, datalen)
        out[low:high] = mw.evaluate(data[low:high], summarize_losses=False)
    return out


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


def pretrained_performance_trainset_selection(mw: ModelWrapper, num_keep_images: int):
    data = load_train(shuffle=False, gray=True)
    losses = build_losses_array(mw, data)

    datalen = len(losses)
    worst_idxes = np.argpartition(losses, kth=datalen-num_keep_images-1)[-num_keep_images:]

    return data[worst_idxes]


if __name__ == '__main__':
    test = load(dataset_path(), force_format=[240, 220, 3])
    data = attach_histogram_to_batch(test, 2)
    print(np.shape(data))
    print(data)
    d2 = shuffle_data((data, data), keep_probability=0.1)
    print(np.shape(d2))
    print(d2)







