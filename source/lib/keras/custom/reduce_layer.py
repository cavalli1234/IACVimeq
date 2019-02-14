from keras import backend as K
from keras.layers import Layer
import tensorflow as tf


class ReduceLayer(Layer):
    def __init__(self, channels, **kwargs):
        self.channels = channels
        super(ReduceLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReduceLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        splits = tf.split(value=inputs, num_or_size_splits=self.channels, axis=-1)
        out_splits = []
        for split in splits:
            out_splits.append(K.sum(split, axis=3))
        return tf.stack(out_splits, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.channels,)