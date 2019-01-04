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
    for _ in range(layers-1):
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

if __name__ == '__main__':
    model = dummy_cnn()
    model.summary()