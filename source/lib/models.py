import keras.layers as kl
import keras.backend as kb
import keras.models as km
import numpy as np
from keras.models import Model
from lib.keras.custom.tile_layer import TileLayer
from lib.keras.custom.reduce_layer import ReduceLayer


def dummy_cnn(channels=3, name='dummy_cnn', activation='relu'):
    model = km.Sequential(name=name)
    model.add(kl.Conv2D(input_shape=(None, None, channels),
                        filters=channels,
                        kernel_size=[3, 3],
                        padding='same'))
    model.add(kl.Activation(activation))
    return model


def ff_hist(n_inputs: int, layers: int = 5, name: str = None):
    bins = n_inputs - 1
    if name is None:
        name = 'ff_L%d_B%d' % (layers, bins)
    model = km.Sequential(name=name)
    model.add(kl.Dense(input_shape=(n_inputs,), units=n_inputs // 2, activation='relu'))
    for _ in range(layers - 3):
        model.add(kl.Dense(units=n_inputs // 2, activation='relu'))
    model.add(kl.Dense(units=n_inputs // 4, activation='tanh'))
    model.add(kl.Dropout(rate=0.85))
    model.add(kl.Dense(units=1, activation='sigmoid'))
    return model


def dummy_ff(input_shape: tuple, name='dummy_ff'):
    in_size = np.prod(input_shape)
    model = km.Sequential(name=name)
    model.add(kl.Flatten(input_shape=input_shape))
    model.add(kl.Dense(units=in_size, activation='sigmoid'))
    model.add(kl.Reshape(target_shape=input_shape))
    return model


def plain_cnn(channels=3, bins=64, layers=1, name='plain_cnn', activation='relu'):
    model = km.Sequential(name='%s_L%d' % (name, layers))
    model.add(kl.InputLayer(input_shape=(None, None, channels)))
    for _ in range(layers - 1):
        model.add(kl.Conv2D(filters=bins,
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
    last_conv = kl.Conv2D(filters=bins,
                          kernel_size=[1, 1],
                          padding='same',
                          activation='relu',
                          kernel_initializer='glorot_normal')(last_conv)
    # adjust and sum up local bins
    for _ in range(layers - 2):
        last_conv = kl.Conv2D(filters=bins,
                              kernel_size=[7, 7],
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


def u_net(channels=1, name='u_net', activation='relu', dropout_rate=0.5):
    inputs = kl.Input(shape=(None, None, channels))

    # Encoding part of the network
    # conv1 = kl.Conv2D(filters=64, kernel_size=[5, 5], padding='same', kernel_initializer='glorot_normal')(inputs)
    # act1 = kl.Activation(activation)(conv1) if type(activation) is str else activation()(conv1)
    # conv2 = kl.Conv2D(filters=64, kernel_size=[5, 5], padding='same', kernel_initializer='glorot_normal')(act1)
    # act2 = kl.Activation(activation)(conv2) if type(activation) is str else activation()(conv2)
    # pool2 = kl.MaxPool2D(pool_size=[2, 2])(act2)

    conv3 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(inputs)
    act3 = kl.Activation(activation)(conv3) if type(activation) is str else activation()(conv3)
    conv3 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(act3)
    act3 = kl.Activation(activation)(conv3) if type(activation) is str else activation()(conv3)
    norm3 = kl.BatchNormalization()(act3)
    pool3 = kl.MaxPool2D(pool_size=[2, 2])(norm3)

    conv4 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(pool3)
    act4 = kl.Activation(activation)(conv4) if type(activation) is str else activation()(conv4)
    conv4 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(act4)
    act4 = kl.Activation(activation)(conv4) if type(activation) is str else activation()(conv4)
    norm4 = kl.BatchNormalization()(act4)
    drop4 = kl.Dropout(rate=dropout_rate)(norm4)
    pool4 = kl.MaxPool2D(pool_size=[2, 2])(drop4)

    conv5 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(pool4)
    act5 = kl.Activation(activation)(conv5) if type(activation) is str else activation()(conv5)
    conv5 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(act5)
    act5 = kl.Activation(activation)(conv5) if type(activation) is str else activation()(conv5)
    drop5 = kl.Dropout(rate=dropout_rate)(act5)

    # Decoding part of the network
    up6 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same',
                    kernel_initializer='glorot_normal')(kl.UpSampling2D(size=[2, 2])(drop5))
    act6 = kl.Activation(activation)(up6) if type(activation) is str else activation()(up6)
    norm6 = kl.BatchNormalization()(act6)
    merge6 = kl.concatenate(inputs=[norm4, norm6])
    conv6 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(merge6)
    act6 = kl.Activation(activation)(conv6) if type(activation) is str else activation()(conv6)
    conv6 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(act6)
    act6 = kl.Activation(activation)(conv6) if type(activation) is str else activation()(conv6)

    up7 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                    kernel_initializer='glorot_normal')(kl.UpSampling2D(size=[2, 2])(act6))
    act7 = kl.Activation(activation)(up7) if type(activation) is str else activation()(up7)
    norm7 = kl.BatchNormalization()(act7)
    merge7 = kl.concatenate(inputs=[norm3, norm7])
    conv7 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(merge7)
    act7 = kl.Activation(activation)(conv7) if type(activation) is str else activation()(conv7)
    conv7 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(act7)
    act7 = kl.Activation(activation)(conv7) if type(activation) is str else activation()(conv7)
    drop7 = kl.Dropout(rate=dropout_rate)(act7)

    # Output with one filter and sigmoid activation function
    out = kl.Conv2D(filters=channels, kernel_size=[1, 1], activation='sigmoid', kernel_initializer='glorot_normal')(
        drop7)

    eta_net_model = Model(inputs=(inputs,), outputs=(out,), name=name)
    return eta_net_model


def hybrid_net(channels=3, bins=128, masks=10, semantic_width=64, dropout_rate=0.5, layers=5, name='hybrid_net'):
    inp = kl.Input(shape=(None, None, channels))
    # add pixel-wise convolutions that should select bins
    last_conv = kl.Conv2D(filters=bins,
                          kernel_size=[1, 1],
                          padding='same',
                          activation='tanh',
                          kernel_initializer='glorot_normal')(inp)
    last_conv = kl.Conv2D(filters=bins,
                          kernel_size=[1, 1],
                          padding='same',
                          activation='relu',
                          kernel_initializer='glorot_normal')(last_conv)
    # adjust and sum up local bins
    for _ in range(layers - 2):
        last_conv = kl.Conv2D(filters=bins,
                              kernel_size=[7, 7],
                              padding='same',
                              activation='relu',
                              kernel_initializer='glorot_normal')(last_conv)
    # attach input image
    conc = kl.Concatenate(axis=-1)([last_conv, inp])

    # infer pixel values pixel-wise
    out_hist = kl.Conv2D(filters=bins//2,
                         kernel_size=[1, 1],
                         padding='same',
                         activation='relu',
                         kernel_initializer='glorot_normal')(conc)
    out_hist = kl.Conv2D(filters=masks*channels,
                         kernel_size=[1, 1],
                         padding='same',
                         activation='sigmoid',
                         kernel_initializer='glorot_normal')(out_hist)

    conv3 = kl.Conv2D(filters=semantic_width//2, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(inp)
    act3 = kl.Activation('relu')(conv3)
    conv3 = kl.Conv2D(filters=semantic_width//2, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(act3)
    act3 = kl.Activation('relu')(conv3)
    norm3 = kl.BatchNormalization()(act3)
    pool3 = kl.MaxPool2D(pool_size=[2, 2])(norm3)

    conv4 = kl.Conv2D(filters=semantic_width, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(pool3)
    act4 = kl.Activation('relu')(conv4)
    conv4 = kl.Conv2D(filters=semantic_width, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(act4)
    act4 = kl.Activation('relu')(conv4)
    norm4 = kl.BatchNormalization()(act4)
    drop4 = kl.Dropout(rate=dropout_rate)(norm4)
    pool4 = kl.MaxPool2D(pool_size=[2, 2])(drop4)

    conv5 = kl.Conv2D(filters=semantic_width, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(pool4)
    act5 = kl.Activation('relu')(conv5)
    conv5 = kl.Conv2D(filters=semantic_width, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(act5)
    act5 = kl.Activation('relu')(conv5)
    drop5 = kl.Dropout(rate=dropout_rate)(act5)

    # Decoding part of the network
    up6 = kl.Conv2D(filters=semantic_width, kernel_size=[3, 3], padding='same',
                    kernel_initializer='glorot_normal')(kl.UpSampling2D(size=[2, 2])(drop5))
    act6 = kl.Activation('relu')(up6)
    norm6 = kl.BatchNormalization()(act6)
    merge6 = kl.concatenate(inputs=[norm4, norm6])
    conv6 = kl.Conv2D(filters=semantic_width, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(merge6)
    act6 = kl.Activation('relu')(conv6)
    conv6 = kl.Conv2D(filters=semantic_width, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(act6)
    act6 = kl.Activation('relu')(conv6)

    up7 = kl.Conv2D(filters=semantic_width//2, kernel_size=[3, 3], padding='same',
                    kernel_initializer='glorot_normal')(kl.UpSampling2D(size=[2, 2])(act6))
    act7 = kl.Activation('relu')(up7)
    norm7 = kl.BatchNormalization()(act7)
    merge7 = kl.concatenate(inputs=[norm3, norm7])
    conv7 = kl.Conv2D(filters=semantic_width//2, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(merge7)
    act7 = kl.Activation('relu')(conv7)
    conv7 = kl.Conv2D(filters=semantic_width//2, kernel_size=[3, 3], padding='same', kernel_initializer='glorot_normal')(act7)
    act7 = kl.Activation('relu')(conv7)
    drop7 = kl.Dropout(rate=dropout_rate)(act7)

    # Output with one filter and sigmoid activation function
    out_unet = kl.Conv2D(filters=masks, kernel_size=[1, 1], activation='sigmoid', kernel_initializer='glorot_normal')(
        drop7)

    # expanded_out_unet = kb.tile(out_unet, [1, 1, 1, channels])
    expanded_out_unet = TileLayer(n_tiles=channels)(out_unet)

    out = kl.Multiply()([expanded_out_unet, out_hist])

    out = ReduceLayer(channels=channels)(out)
    model = km.Model(name='%s_L%d_B%d_S%d_M%d' % (name, layers, bins, semantic_width, masks),
                     inputs=[inp],
                     outputs=[out])

    return model


if __name__ == '__main__':
    model = u_net(channels=3)
    model.summary()
