import sys
import getopt
from lib.model_wrap.model_wrapper import ModelWrapper
from lib.model_wrap.pixwise_model_wrapper import PixwiseModelWrapper
from lib.models import hist_building_cnn, ff_hist
from utils.logging import log, ERRORS
from utils.utility import *
from keras.losses import mean_squared_error as mse


DEFAULT_OPTS = {
    't': 100,   # training samples
    'v': 10,    # validation samples
    'm': 'conv',  # model type (convolution or pixelwise)
    'i': None,  # input model name (determines preloading of weights)
    'o': None,  # output model name
    'l': 5,     # number of layers for convnet
    'b': 64,    # number of bins to consider
    'k': 0.25,  # keep probability in ff pixel selection
    's': False,  # selective train data selection
    'c': False  # the model is to be loaded from a ckp file
}

DEFAULT_MODELS = {
    'conv': lambda b, l: ModelWrapper(model_generator=lambda: hist_building_cnn(layers=l, bins=b)),
    'ff': lambda b, l: PixwiseModelWrapper(model_generator=lambda: ff_hist(n_inputs=b+1, layers=l))
}


def parse_opts(optlist=sys.argv[1:]):
    in_opts, args = getopt.getopt(optlist, 't:v:m:i:o:l:k:b:sc')
    out_opts = DEFAULT_OPTS
    for (o, v) in in_opts:
        o = re.sub('^-*', '', o)
        if o in 'tvlb':
            out_opts[o] = int(v)
        elif o in 'mio':
            out_opts[o] = v
        elif o in 'k':
            out_opts[o] = float(v)
        elif o in 'sc':
            out_opts[o] = True

    return out_opts


def pretrained_model(opts):
    return opts['i'] is not None


def load_model(opts):
    if not pretrained_model(opts):
        # no old model!
        return DEFAULT_MODELS[opts['m']](opts['b'], opts['l'])

    if opts['c']:
        ext = '.ckp'
        model_generator = lambda: DEFAULT_MODELS[opts['m']](opts['b'], opts['l']).model
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
