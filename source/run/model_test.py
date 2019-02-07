import os.path as op
import sys

sys.path.append(op.realpath(op.join(op.split(__file__)[0], "..")))

from lib.model_wrap.model_wrapper import ModelWrapper
from lib.model_wrap.pixwise_model_wrapper import PixwiseModelWrapper
from lib.models import *
from data.loading import *

if __name__ == '__main__':
    # cnn = ModelWrapper(h5_name='plain_cnn_L5.h5',
    #                    model_generator=lambda: plain_cnn(layers=5))

    # test_rgb = load_valid(500, shuffle=False, gray=False)
    # cnn_loss = cnn.evaluate(test_rgb, plots=4)

    test_gray_v = load_valid(1000, shuffle=False, gray=True)
    test_gray_t = load_train(1000, shuffle=False, gray=True)

    ff = PixwiseModelWrapper(h5_name='ff_hist.h5',
                             #model_generator=lambda: ff_hist(129)
                             )
    ff_loss = ff.evaluate(test_gray_v, plots=4)
    ff_tloss = ff.evaluate(test_gray_t, plots=4)

    # conv_hist = ModelWrapper(h5_name='hist_building_cnn_L5_B128.h5',
    #                         model_generator=lambda: hist_building_cnn(layers=5, bins=128))
    # conv_hist_loss = conv_hist.evaluate(test_gray, plots=4)

    print("Pixwise FF loss: %f" % ff_loss)
    print("Pixwise FF tloss: %f" % ff_loss)
    # print("CNN loss: %f" % cnn_loss)
    # print("CNN_hist loss: %f" % conv_hist_loss)

