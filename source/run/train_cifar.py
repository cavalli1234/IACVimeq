import os.path as op
import sys

sys.path.append(op.realpath(op.join(op.split(__file__)[0], "..")))

from lib.training import train_model
from utils.logging import *
from utils.utility import *
from utils.train_setup import parse_opts, \
    load_data, load_model, setup_train_configuration


def main():
    optlist = sys.argv[1:]
    optlist = '-m unet -c 1 -t 3 -v 1'.split()
    opts = parse_opts(optlist)
    model = load_model(opts)
    train, valid = load_data(opts, model)
    config = setup_train_configuration(opts=opts,
                                       mw=model,
                                       train=train,
                                       valid=valid)
    train_model(**config)


if __name__ == '__main__':
    set_verbosity(DEBUG)
    main()

