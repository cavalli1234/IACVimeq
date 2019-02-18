import os.path as op
import sys

sys.path.append(op.realpath(op.join(op.split(__file__)[0], "..")))

from data.loading import *
from utils.train_setup import parse_opts, load_model

def main():
    optlist = sys.argv[1:]
    # optlist = '-i plain_cnn_L10 -m plain -l 10 -b 128 -c 1 -t 1 -v 1 --from-fresh'.split()
    opts = parse_opts(optlist)
    model = load_model(opts)
    opts['s'] = False
    # train = load_train(opts['t'], shuffle=False)
    valid = load_valid(opts['v'])
    # trainloss = model.evaluate(train, plots=4)
    validloss = model.evaluate(valid, plots=4, save_png=opts['plot'])

    # print(opts['i'], " train loss: ", trainloss)
    print(opts['i'], " valid loss: ", validloss)


if __name__ == '__main__':
    main()

