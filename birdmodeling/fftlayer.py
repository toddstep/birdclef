import math

import tensorflow as tf
from keras import layers
import tensorflow_io as tfio

import constants

# https://www.tensorflow.org/guide/keras/custom_layers_and_models
# https://medium.com/p/2d9d91360f06
num_unused_ceps = 21


# https://www.tensorflow.org/io/tutorials/audio
class FFT(layers.Layer):
    """Spectrogram of signal

    Args:
        frame_sec (float): (BASE) Width (in seconds) of analysis window pyramids
        stride_sec (float): Shift (in seconds) between analysis windows
        freq_drop_rate (float):
        top_db (float):
        freq_mask_param (int):
        time_mask_param (int):
        gauss_noise_param (float):
        reduce_db_param (float):
        do_freq_fade (bool):
        do_time_fade (bool):
    """
    def __init__(self, frame_sec=0.060, stride_sec=0.01675, freq_drop_rate=None, top_db=80., freq_mask_param=None, time_mask_param=None, gauss_noise_param=None, reduce_db_param=None, do_freq_fade=False, do_time_fade=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_sec = frame_sec
        self.stride_sec = stride_sec
        self.n_fft = int(self.frame_sec * constants.model_sr)
        self.window_pyr = [int(self.n_fft * x) for x in [2, .25, .0625]]
        self.stride = int(self.stride_sec * constants.model_sr)
        self.num_channels = 3
        if freq_drop_rate is not None:
            assert 0 <= freq_drop_rate <= 1, "Drop rate must be None or in [0, 1]"
        self.freq_drop_rate = freq_drop_rate
        self.top_db = top_db
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.gauss_noise_param = gauss_noise_param
        self.reduce_db_param = reduce_db_param
        self.do_freq_fade=do_freq_fade
        self.do_time_fade=do_time_fade
        
    def call(self, inputs, training=False):
        spec = tf.stack([self._spec(inputs, window) for window in self.window_pyr], axis=-1)
        print("CALL",training)
        if training:
            if self.gauss_noise_param is not None and self.gauss_noise_param>0:
                spec = tf.map_fn(self._add_gauss_noise, spec)
        spec = tfio.audio.dbscale(spec, top_db=self.top_db)
        # https://stackoverflow.com/questions/38376478/changing-the-scale-of-a-tensor-in-tensorflow
        min0, max0 = tf.reduce_min(spec), tf.reduce_max(spec)
        spec = (spec-min0)/(max0-min0)
        if training:
            if self.reduce_db_param is not None and self.reduce_db_param>0:
                spec = tf.map_fn(self._reduce_top_db, spec)
            if self.freq_mask_param is not None and self.freq_mask_param>0:
                spec = tf.map_fn(self._freq_mask, spec)
            if self.time_mask_param is not None and self.time_mask_param>0:
                spec = tf.map_fn(self._time_mask, spec)
            if self.do_freq_fade:
                spec = tf.map_fn(FFT._freq_fade, spec)
            if self.do_time_fade:
                spec = tf.map_fn(FFT._time_fade, spec)
            if self.freq_drop_rate is not None and self.freq_drop_rate>0:
                spec = self._drop(spec)
        return 2*spec-1

    def _spec(self, inputs, window):
        spec = tfio.audio.spectrogram(inputs, nfft=self.n_fft, window=window, stride=self.stride)
        spec = tfio.audio.melscale(spec,
                                   rate=constants.model_sr,
                                   mels=(num_unused_ceps+299),
                                   fmin=0,
                                   fmax=(constants.model_sr // 2)
                                  )
        spec = spec[..., num_unused_ceps:]
        return spec

    def _drop(self, spec):
        rand_channel = tf.random.uniform(tf.shape(spec)[:-1])[..., tf.newaxis]
        rand_spec = tf.repeat(rand_channel, 3, axis=-1)
        drop_ind = tf.where(rand_spec < self.freq_drop_rate)
        drop_vals = tf.tile([0.], tf.shape(drop_ind)[:1])
        spec = tf.tensor_scatter_nd_update(spec, drop_ind, drop_vals)
        return spec

    def _freq_mask(self, spec):
        return FFT._mask(spec, self.freq_mask_param, tfio.audio.freq_mask)

    def _time_mask(self, spec):
        return FFT._mask(spec, self.time_mask_param, tfio.audio.time_mask)

    def _reduce_top_db(self, spec):
        new_top = tf.random.uniform((), 0., self.reduce_db_param)
        spec = tf.where(spec<new_top, new_top*tf.ones_like(spec), spec)
        spec = (spec - new_top) / (1-new_top)
        return spec

    def _add_gauss_noise(self, spec):
        stddev = tf.random.uniform((), 0., self.gauss_noise_param)
        noise = tf.random.normal(spec.shape, mean=0., stddev=stddev, name='GaussNoise')
        return spec+noise

    @staticmethod
    def _freq_fade(spec, mode='logarithmic'):
        perm = [1, 0, 2]
        spec = tf.transpose(spec, perm=perm)
        return tf.transpose(FFT._fade(spec, mode), perm=perm)

    @staticmethod
    def _time_fade(spec, mode='logarithmic'):
        return FFT._fade(spec, mode)

    @staticmethod
    def _fade(spec, mode):
        max_fade = spec.shape[0]
        fade_in = tf.random.uniform((), 0, max_fade, dtype=tf.int32)
        fade_out = tf.random.uniform((), 0, max_fade, dtype=tf.int32)
        spec = tfio.audio.fade(spec, fade_in=fade_in, fade_out=fade_out, mode=mode)
        return spec

    @staticmethod
    def _mask(spec, param, mask_func):
        mask = tf.ones_like(spec[..., 0])
        mask = mask_func(mask, param)
        mask = tf.repeat(mask[...,tf.newaxis], 3, -1)
        return spec*mask
