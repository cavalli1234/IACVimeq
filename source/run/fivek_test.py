import os.path as op
import sys

sys.path.append(op.realpath(op.join(op.split(__file__)[0], "..")))

from utils.train_setup import parse_opts, load_model, load_data_expert

def main():
    optlist = sys.argv[1:]
    # optlist = '-i hist_building_cnn_L5_B64_fivek -m hist -c 3'.split()
    opts = parse_opts(optlist)
    model = load_model(opts)
    opts['s'] = False
    train, valid = load_data_expert(opts, model)
    trainloss = model.evaluate(train[0], gt_batch=train[1], plots=4)
    validloss = model.evaluate(valid[0], gt_batch=valid[1], plots=4)

    print(opts['i'], " train loss: ", trainloss)
    print(opts['i'], " valid loss: ", validloss)


if __name__ == '__main__':
    main()

