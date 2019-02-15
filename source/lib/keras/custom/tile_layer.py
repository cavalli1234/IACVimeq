from keras import backend as K
from keras.layers import Layer


class TileLayer(Layer):
    def __init__(self, n_tiles=3, **kwargs):
        self.n_tiles = n_tiles
        super(TileLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TileLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        return K.tile(K.softmax(inputs), [1, 1, 1, self.n_tiles])

    def compute_output_shape(self, input_shape):
        out_shape = list(input_shape)
        out_shape[-1] = out_shape[-1] * self.n_tiles
        return tuple(out_shape)