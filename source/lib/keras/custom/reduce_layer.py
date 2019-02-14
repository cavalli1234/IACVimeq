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
        return tf.math.add_n(splits)

    def compute_output_shape(self, input_shape):
        out_shape = list(input_shape)
        out_shape[-1] = out_shape[-1] // self.channels
        return tuple(out_shape)