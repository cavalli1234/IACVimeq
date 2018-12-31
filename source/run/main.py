from data.img_io import load
from matplotlib.pyplot import imshow, show
from utils.naming import *
from skimage import exposure
from lib.models import dummy_cnn
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
        out[idx] = exposure.equalize_hist(imgs[idx])
    return out


def cut_left(data, split):
    return np.array(data[:int(len(data)*split)])


def cut_right(data, split):
    return np.array(data[int(len(data)*split):])


if __name__ == '__main__':
    set_verbosity(DEBUG)
    imgs = load(dataset_path(), force_format=[840, 760, 3])
    gt = make_ground_truth(imgs)

    valid_split = 0.5

    train = (cut_left(imgs, valid_split), cut_left(gt, valid_split))
    valid = (cut_right(imgs, valid_split), cut_right(gt, valid_split))

    model = train_model(model_generator=dummy_cnn,
                        train=train,
                        valid=valid,
                        loss=mse,
                        patience=5)

    showimgs(valid[0][0:2])
    showimgs(model.predict(valid[0])[0:2])
    showimgs(valid[1][0:2])

