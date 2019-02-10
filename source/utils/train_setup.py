import sys
import getopt
from lib.model_wrap.model_wrapper import ModelWrapper
from lib.model_wrap.pixwise_model_wrapper import PixwiseModelWrapper
from lib.models import hist_building_cnn, ff_hist
from utils.logging import log, ERRORS, IMPORTANT_WARNINGS
from utils.utility import *
from keras.losses import mean_squared_error as mse
from utils.naming import fivek_element, fivek_dimension
from data.img_io import load


DEFAULT_OPTS = {
    't': 100,   # training samples
    'v': 10,    # validation samples
    'm': 'conv',  # model type (convolution or pixelwise)
    'i': None,  # input model name (determines preloading of weights)
    'o': None,  # output model name
    'l': 5,     # number of layers for convnet
    'b': 64,    # number of bins to consider
    'c': 1,     # number of channels of images (1 gray or 3 rgb)
    'k': 0.25,  # keep probability in ff pixel selection
    's': False,  # selective train data selection
    'ckp': False,  # the model is to be loaded from a ckp file
    'e': 1      # target expert for ground truth
}

DEFAULT_MODELS = {
    'conv': lambda c, b, l: ModelWrapper(model_generator=lambda: hist_building_cnn(channels=c, layers=l, bins=b)),
    'ff': lambda c, b, l: PixwiseModelWrapper(model_generator=lambda: ff_hist(n_inputs=b+1, layers=l))
}


def parse_opts(optlist=sys.argv[1:]):
    in_opts, args = getopt.getopt(optlist, 't:v:m:i:o:l:k:b:sc:',
                                  longopts=['ckp'])
    out_opts = DEFAULT_OPTS
    for (o, v) in in_opts:
        o = re.sub('^-*', '', o)
        if o == 'ckp':
            out_opts[o] = True
        elif o in 'tvlbc':
            out_opts[o] = int(v)
        elif o in 'mio':
            out_opts[o] = v
        elif o in 'k':
            out_opts[o] = float(v)
        elif o in 's':
            out_opts[o] = True

    return out_opts


def pretrained_model(opts):
    return opts['i'] is not None


def load_model(opts):
    if not pretrained_model(opts):
        # no old model!
        return DEFAULT_MODELS[opts['m']](opts['c'], opts['b'], opts['l'])

    if opts['ckp']:
        ext = '.ckp'
        model_generator = lambda: DEFAULT_MODELS[opts['m']](opts['c'], opts['b'], opts['l']).model
    else:
        ext = '.h5'
        model_generator = None

    if opts['m'] == 'conv':
        mw = ModelWrapper(model_file=opts['i'] + ext, model_generator=model_generator)
    elif opts['m'] == 'ff':
        mw = PixwiseModelWrapper(model_file=opts['i'] + ext, model_generator=model_generator)
    else:
        log("Options -m "+opts['m']+" unrecognized. Use -m ff or -m conv",
            level=ERRORS)
        raise ValueError()
    if opts['o'] is not None:
        mw.model.name = opts['o']
    return mw


def load_data(opts: dict, mw: ModelWrapper):
    TRAIN_SAMPLES = opts['t']
    VALID_SAMPLES = opts['v']

    def pretrained_train_selection():
        return pretrained_performance_trainset_selection(mw, num_keep_images=TRAIN_SAMPLES)

    def basic_train_selection():
        return load_train(TRAIN_SAMPLES, gray=True)

    train = pretrained_train_selection() if pretrained_model(opts) and opts['s'] else basic_train_selection()
    valid = load_valid(VALID_SAMPLES, gray=True)

    def ff_preprocess():
        return preprocess_data_ffnn(train, valid, bins=opts['b'], kp=opts['k'])

    def conv_preprocess():
        return preprocess_data_cnn(train, valid)

    train, valid = conv_preprocess() if opts['m'] == 'conv' else ff_preprocess()

    return train, valid


def load_data_expert(opts: dict, mw: ModelWrapper):
    TRAIN_SAMPLES = opts['t']
    VALID_SAMPLES = opts['v']

    TOT_SAMPLES = TRAIN_SAMPLES+VALID_SAMPLES

    if fivek_dimension() < TOT_SAMPLES:
        log("Warning: required %d samples but only %d are avalable." % (TOT_SAMPLES, fivek_dimension()),
            IMPORTANT_WARNINGS)

    train_idxs = list(range(TRAIN_SAMPLES))
    valid_idxs = list(range(TRAIN_SAMPLES, TOT_SAMPLES))

    train = load(path=fivek_element(idx=train_idxs),
                 force_major_side_x=True,
                 force_format=(500, 333, 3))
    train_gt = load(path=fivek_element(idx=train_idxs, expert=opts['e']),
                    force_major_side_x=True,
                    force_format=(500, 333, 3))

    valid = load(path=fivek_element(idx=valid_idxs),
                 force_major_side_x=True,
                 force_format=(500, 333, 3))
    valid_gt = load(path=fivek_element(idx=valid_idxs, expert=opts['e']),
                    force_major_side_x=True,
                    force_format=(500, 333, 3))

    return (train, train_gt), (valid, valid_gt)


def setup_train_configuration(opts, mw, train, valid):
    return {
        'model_generator': lambda: mw.model,
        'train': train,
        'valid': valid,
        'loss': mse,
        'patience': 5,
        'learning_rate': 1e-4,
        'max_epochs': 200,
        'log_images': opts['m'] == 'conv'
    }
