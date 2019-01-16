import keras.models as km
from skimage import exposure
import numpy as np
from utils.naming import *
import matplotlib.pyplot as plt


def make_ground_truth(imgs: np.ndarray):
    out = np.zeros_like(imgs)
    for idx in range(len(imgs)):
        tmp = exposure.equalize_hist(imgs[idx])
        # returned equalized image is in 0-1 floating points!
        if out.dtype == np.uint8:
            tmp *= 255
        out[idx] = tmp
    return out


class ModelWrapper:
    def __init__(self, h5_name: str, model_generator=None):
        if model_generator is None:
            self.model: km.Model = km.load_model(models_path(h5_name))
        else:
            self.model: km.Model = model_generator()
            self.model.load_weights(models_path(h5_name))

        self.need_reshape_out = False

    def preprocess_input(self, inp):
        if len(self.model.input_shape) > np.ndim(inp):
            inp = np.reshape(inp, newshape=np.shape(inp)+(1,))
            self.need_reshape_out = True
        else:
            self.need_reshape_out = False
        return inp

    def postprocess_output(self, out):
        if self.need_reshape_out:
            out = np.reshape(out, newshape=np.shape(out)[:-1])
        return out

    def predict(self, in_imgs):
        return self.postprocess_output(self.model.predict(self.preprocess_input(in_imgs)))

    def evaluate(self, in_batch, gt_batch=None, plots=0):
        if gt_batch is None:
            gt_batch = make_ground_truth(in_batch)
        pred_batch = self.predict(in_batch)
        diff_batch = [np.square(gt_batch[idx]-pred_batch[idx]) for idx in range(len(gt_batch))]
        losses_array = np.mean(diff_batch, axis=tuple(range(1, len(np.shape(diff_batch)))))
        if plots > 0:
            plots = min(plots, len(in_batch))
            vert_concat = lambda x: np.concatenate(tuple(x[:plots]), axis=0)

            norm_diff_batch = [d/np.max(d) for d in diff_batch[:plots]]

            in_plots = vert_concat(in_batch)
            gt_plots = vert_concat(gt_batch)
            pr_plots = vert_concat(pred_batch)
            df_plots = vert_concat(norm_diff_batch)

            total = np.concatenate((in_plots, gt_plots, pr_plots, df_plots), axis=1)

            cmap = None
            # handle grayscale
            if np.shape(total)[-1] == 1:
                total = np.reshape(total, np.shape(total)[:-1])
                cmap = 'gray'
            elif np.ndim(total) == 2:
                cmap = 'gray'

            plt.title("Image samples")
            plt.imshow(total, cmap=cmap, vmin=0, vmax=1)
            plt.show()
            plt.title("Imagewise loss distribution")
            plt.hist(losses_array, bins=100)
            plt.show()
            # plt.title("Pixelwise loss distribution")
            # plt.hist(np.reshape(diff_batch, newshape=(np.size(diff_batch),)), bins=2048)
            # plt.show()
        return np.mean(losses_array)


if __name__ == '__main__':
    from lib.models import *
    from data.loading import *

    mw = ModelWrapper(h5_name='plain_cnn_L5.h5',
                      model_generator=lambda: plain_cnn(layers=5))
    test = load_valid(1000, gray=False)
    print(mw.evaluate(test, plots=3))