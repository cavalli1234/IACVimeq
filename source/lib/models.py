import keras.layers as kl
import keras.models as km


def dummy_cnn(channels=3, name='dummy_cnn', activation='relu'):
    model = km.Sequential(name=name)
    model.add(kl.Conv2D(input_shape=(None, None, channels),
                        filters=channels,
                        kernel_size=[3, 3],
                        padding='same'))
    model.add(kl.Activation(activation))
    return model


if __name__ == '__main__':
    model = dummy_cnn(weight_decay=10)
    model.summary()