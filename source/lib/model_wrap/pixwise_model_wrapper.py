from lib.model_wrap.model_wrapper import ModelWrapper
from utils.utility import attach_histogram
import numpy as np


def _batch_attach_histogram(batch_gray, nbins: int = 256, normalize: bool = True):
    data = [attach_histogram(x, nbins, normalize) for x in batch_gray]
    result = np.reshape(np.array(data), (np.size(data)//(nbins+1), nbins+1))
    return result


class PixwiseModelWrapper(ModelWrapper):
    def __init__(self, h5_name: str, model_generator=None):
        ModelWrapper.__init__(self, h5_name=h5_name,
                              model_generator=model_generator)
        self.last_input_shape = None
        self.nbins = self.model.input_shape[1]-1

    def preprocess_input(self, inp):
        self.last_input_shape = np.shape(inp)[1:]
        return _batch_attach_histogram(inp, self.nbins)

    def postprocess_output(self, out):
        return np.reshape(out, newshape=(np.size(out)//np.prod(self.last_input_shape),)+self.last_input_shape)


if __name__ == '__main__':
    from lib.models import *
    from data.img_io import load
    from utils.naming import *
    from keras.datasets import cifar10
    from skimage import color

    mw = PixwiseModelWrapper(h5_name='ff_hist.h5')
    # test = load(dataset_path(), force_format=[240, 220, 1])
    (_, _), (test, _) = cifar10.load_data()
    np.random.shuffle(test)
    test = test[:100] / 255.0
    test = [color.rgb2gray(t) for t in test]
    print(mw.evaluate(test, plots=4))