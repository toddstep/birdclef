import librosa
import math
import matplotlib.pyplot as plt
import os
import pickle
import random

import tensorflow as tf
import tensorflow_io as tfio

import constants

def tensor_string(x):
    if isinstance(x, tf.Tensor):
        return x.numpy().decode()
    else:
        return x
    
def ogg_fn(filename):
    return os.path.join(constants.base_dir, 'train_audio', tensor_string(filename))

def duration(x):
    return librosa.get_duration(path=ogg_fn(tf.constant(x)))

def load_audio_tensor(filename, primary_label):
    # https://www.tensorflow.org/tutorials/audio/transfer_learning_audio
    # in_sr = librosa.get_samplerate(ogg_fn(filename))
    filename_ogg = tf.py_function(ogg_fn, [filename], tf.string)
    x = tf.io.read_file(filename_ogg)
    x = tfio.audio.decode_vorbis(x)
    # x = tfio.audio.decode_mp3(x)
    # x = tfio.audio.resample(x, in_sr, constants.model_sr)
    x = x[:, 0]
    return x, primary_label

def shift_audio(audio, label):
    shift = tf.random.uniform((), 0, len(audio), dtype=tf.int32)
    # shift = tf.py_function(lambda x: random.randint(0, x), [len(audio)], tf.int32)
    audio_shifted = tf.concat([audio[shift:], audio[:shift]], axis=0)    
    return audio_shifted, label

def match_lengths(orig_audio, other_audio):
    len_orig = len(orig_audio)
    if len(other_audio) < len_orig:
        num_copies = tf.math.ceil(len_orig / len(other_audio))
        num_copies = tf.cast(num_copies, tf.int32)
        other_audio = tf.tile(other_audio, (num_copies,))
    other_audio = other_audio[:len_orig]
    return other_audio

def combine_recordings(orig_audio, label, index_filename_list, weight=None):
    len_orig = len(orig_audio)
    
    other_file = tf.py_function(lambda x: random.choices(index_filename_list[x]),
                              [label],
                              tf.string)
    other_audio, other_label = load_audio_tensor(other_file, label)
    other_audio = match_lengths(orig_audio=orig_audio, other_audio=other_audio)
    
    if weight is None:
        weight = tf.random.uniform((), .5, 1.)
    # weight = tf.py_function(lambda x: random.uniform(x, 1.), [tf.constant(.5)], tf.float32)
    # augment_audio = weight * orig_audio + (1-weight) * other_audio
    augment_audio = tf.reduce_sum([weight*orig_audio, (1-weight)*other_audio],axis=0)

    return augment_audio, label

def frame_audio(audio, label):
    num_samples = len(audio)
    num_repeats = tf.py_function(math.ceil, [constants.fixed_num_samples/num_samples], tf.int32)
    if num_repeats > 1:
        audio = tf.tile(audio, (num_repeats,) )
    if len(audio)>constants.fixed_num_samples:
        audio = audio[:constants.fixed_num_samples]
    audio = tf.signal.frame(audio, frame_length=constants.frame_length, frame_step=constants.frame_step)
    return audio, label
