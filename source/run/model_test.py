import os.path as op
import sys

sys.path.append(op.realpath(op.join(op.split(__file__)[0], "..")))

from lib.model_wrap.model_wrapper import ModelWrapper
from lib.model_wrap.pixwise_model_wrapper import PixwiseModelWrapper
from lib.models import *
from data.loading import *
from utils.train_setup import parse_opts, load_model, load_data

def main():
    optlist = sys.argv[1:]
    # optlist = '-i ff_hist -m ff'.split()
    opts = parse_opts(optlist)
    model = load_model(opts)
    opts['s'] = False
    train = load_train(500, shuffle=False)
    valid = load_valid(500)
    trainloss = model.evaluate(train, plots=4)
    validloss = model.evaluate(valid, plots=4)

    print(opts['i'], " train loss: ", trainloss)
    print(opts['i'], " valid loss: ", validloss)


if __name__ == '__main__':
    main()

