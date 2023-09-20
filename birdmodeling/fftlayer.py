import math

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_io as tfio

import constants

# https://www.tensorflow.org/guide/keras/custom_layers_and_models
# https://medium.com/p/2d9d91360f06

class FFT(layers.Layer):
    """Spectrogram of signal

    Args:
        max_freq (int): Max frequency in spectrogram
        frame_sec (float): Width (in seconds) of analysis window
        stride_sec (float): Shift (in seconds) between analysis windows
    """
    def __init__(self, max_freq=8000, frame_sec=.025, stride_sec=.01):
        super().__init__()
        self.max_freq = max_freq
        self.frame_sec = frame_sec
        self.stride_sec = stride_sec
        self.window = int(self.frame_sec * constants.model_sr)
        # self.n_fft = 2**math.ceil(math.log(self.window, 2)) 
        self.n_fft = self.window
        self.stride = int(self.stride_sec * constants.model_sr)
        self.max_keep_freq = math.ceil(self.max_freq / constants.model_sr * self.n_fft)
        self.num_channels = 3
        
    def _spec(self, inputs):
        num_samples = inputs.shape[-1]
        spec = tfio.audio.spectrogram(inputs, nfft=self.n_fft, window=self.window, stride=self.stride)
        spec = tf.math.log(spec[..., :self.max_keep_freq])
        return spec
    
    def _norm_spec(self, spec):
        spec = tf.math.maximum(spec, -5)
        spec_max = tf.reduce_max(spec)
        spec_min = tf.reduce_min(spec)
        spec = (spec-spec_min)/(spec_max-spec_min)
        spec = 2*spec - 1
        return spec
    
    def call(self, inputs):
        spec = self._spec(inputs)
        spec = self._norm_spec(spec)
        spec = spec[..., tf.newaxis]
        spec = tf.repeat(spec, self.num_channels, axis=-1)
        return spec
