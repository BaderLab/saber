# -*- coding: utf-8 -*-
import keras.backend as K
from keras.layers import Dropout
from keras.engine import InputSpec

class TimestepDropout(Dropout):
    """Timestep Dropout.

    This version performs the same function as Dropout, however it drops entire
    timesteps (e.g., words embeddings in NLP tasks) instead of individual
    elements (features).

    Arguments
        rate (int): float between 0 and 1. Fraction of the timesteps to drop.

    # Input shape
        3D tensor with shape: `(samples, timesteps, channels)`

    # Output shape
        Same shape as the input.

    References
        - A Theoretically Grounded Application of Dropout in Recurrent Neural
        Networks (https://arxiv.org/pdf/1512.05287)
        - GitHub issue (https://github.com/keras-team/keras/issues/7290)
    """

    def __init__(self, rate, **kwargs):
        super(TimestepDropout, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape
