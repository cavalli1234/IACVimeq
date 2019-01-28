import keras.layers as kl
import keras.models as km
import numpy as np


def dummy_cnn(channels=3, name='dummy_cnn', activation='relu'):
    model = km.Sequential(name=name)
    model.add(kl.Conv2D(input_shape=(None, None, channels),
                        filters=channels,
                        kernel_size=[3, 3],
                        padding='same'))
    model.add(kl.Activation(activation))
    return model


def ff_hist(n_inputs: int, name: str = 'ff_hist'):
    model = km.Sequential(name=name)
    model.add(kl.Dense(input_shape=(n_inputs,), units=n_inputs//2, activation='relu'))
    model.add(kl.Dense(units=n_inputs//2, activation='relu'))
    model.add(kl.Dropout(rate=0.8))
    model.add(kl.Dense(units=1, activation='sigmoid'))
    return model


def dummy_ff(input_shape: tuple, name='dummy_ff'):
    in_size = np.prod(input_shape)
    model = km.Sequential(name=name)
    model.add(kl.Flatten(input_shape=input_shape))
    model.add(kl.Dense(units=in_size, activation='sigmoid'))
    model.add(kl.Reshape(target_shape=input_shape))
    return model


def plain_cnn(channels=3, layers=1, name='plain_cnn', activation='relu'):
    model = km.Sequential(name='%s_L%d' % (name, layers))
    model.add(kl.InputLayer(input_shape=(None, None, channels)))
    for _ in range(layers - 1):
        model.add(kl.Conv2D(filters=64,
                            kernel_size=[3, 3],
                            padding='same',
                            activation=activation,
                            kernel_initializer='glorot_normal'))
    model.add(kl.Conv2D(filters=channels,
                        kernel_size=[3, 3],
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='glorot_normal'))
    return model


def hist_building_cnn(channels=1, layers=1, bins=128, name='hist_building_cnn', activation='relu'):
    inp = kl.Input(shape=(None, None, channels))
    # add pixel-wise convolutions that should select bins
    last_conv = kl.Conv2D(filters=bins,
                          kernel_size=[1, 1],
                          padding='same',
                          activation='tanh',
                          kernel_initializer='glorot_normal')(inp)
    """
    last_conv = kl.Conv2D(filters=bins,
                          kernel_size=[1, 1],
                          padding='same',
                          activation='relu',
                          kernel_initializer='glorot_normal')(last_conv)
    """
    # adjust and sum up local bins
    for _ in range(layers - 1):
        last_conv = kl.Conv2D(filters=bins,
                              kernel_size=[3, 3],
                              padding='same',
                              activation=activation,
                              kernel_initializer='glorot_normal')(last_conv)
    # attach input image
    conc = kl.Concatenate(axis=-1)([last_conv, inp])

    # infer pixel values pixel-wise
    out = kl.Conv2D(filters=64,
                    kernel_size=[1, 1],
                    padding='same',
                    activation=activation,
                    kernel_initializer='glorot_normal')(conc)
    out = kl.Conv2D(filters=channels,
                    kernel_size=[1, 1],
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='glorot_normal')(out)

    model = km.Model(name='%s_L%d_B%d' % (name, layers, bins),
                     inputs=[inp],
                     outputs=[out])
    return model


if __name__ == '__main__':
    model = hist_building_cnn(layers=3, bins=128)
    model.summary()
