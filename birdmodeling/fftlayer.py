import math

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_io as tfio

import constants

# https://www.tensorflow.org/guide/keras/custom_layers_and_models
# https://medium.com/p/2d9d91360f06
num_unused_ceps = 21
min_melspec_value = -100.


class FFT(layers.Layer):
    """Spectrogram of signal

    Args:
        frame_sec (float): Width (in seconds) of analysis window
        stride_sec (float): Shift (in seconds) between analysis windows
    """
    def __init__(self, frame_sec=0.060, stride_sec=0.01675, drop_rate=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_sec = frame_sec
        self.stride_sec = stride_sec
        self.window = int(self.frame_sec * constants.model_sr)
        self.n_fft = self.window
        self.stride = int(self.stride_sec * constants.model_sr)
        self.num_channels = 3
        if drop_rate is not None:
            assert 0 < drop_rate <= 1, "Drop rate must be None or in (0, 1]"
        self.drop_rate = drop_rate
        
    def _spec(self, inputs):
        spec = tfio.audio.spectrogram(inputs, nfft=self.n_fft, window=self.window, stride=self.stride)
        return spec
    
    def _norm_spec(self, spec):
        spec = tf.math.maximum(spec, min_melspec_value)
        # spec_max = tf.reduce_max(spec)
        # spec_min = tf.reduce_min(spec)
        # spec = (spec-spec_min)/(spec_max-spec_min)
        # spec = 2*spec - 1
        return spec

    def _drop(self, spec):
        drop_ind = tf.where(tf.random.uniform(tf.shape(spec)) < self.drop_rate)
        drop_vals = tf.tile([min_melspec_value], tf.shape(drop_ind)[:1])
        spec = tf.tensor_scatter_nd_update(spec, drop_ind, drop_vals)
        return spec
    
    def call(self, inputs, training=False):
        # centering = (self.n_fft // 2) - self.stride
        # inputs = tf.pad(inputs, tf.constant([[centering, 0]]), 'REFLECT')
        spec = self._spec(inputs)
        spec = tf.math.pow(spec, 2)
        spec = tfio.audio.melscale(spec,
                                   rate=constants.model_sr,
                                   mels=(num_unused_ceps+299),
                                   fmin=0,
                                   fmax=(constants.model_sr // 2)
                                  )
        spec = tf.math.log(spec)
        spec = spec[..., num_unused_ceps:]
        spec = self._norm_spec(spec)
        spec = spec[..., tf.newaxis]
        spec = tf.repeat(spec, self.num_channels, axis=-1)
        # spec = tf.repeat(spec, self.num_channels, axis=3)
        if training and self.drop_rate is not None:
            spec = self._drop(spec)
        return spec
