import keras

import librosa
import math
import numpy as np
import os
import pickle
import random

import tensorflow as tf
import tensorflow_io as tfio

import constants

class LoadAudio(keras.Model):
    def __init__(self, augment=False):
        super().__init__()
        self.augment = augment

    def call(self, inputs):
        # AudioIOTensor() has buggy output when dataset does parallel maps (and iterator is restarted multiple times)
        # x = tfio.audio.AudioIOTensor(inputs, dtype='float32').to_tensor()
        x = tf.io.read_file(inputs)
        x = tfio.audio.decode_vorbis(x)

        x = tf.reduce_mean(x, axis=1)
        # audio_start, audio_stop = tfio.audio.trim(x, epsilon=tf.reduce_max(tf.abs(x))*1e-5, axis=0)
        # x = x[audio_start:audio_stop]
        x, _ = clean_audio(x, -1)
        if self.augment:
            x, _ = chop_audio(x, -1)
        return x

load_audio_augment = LoadAudio(augment=True)
load_audio = LoadAudio(augment=False)

# def tensor_string(x):
#     if isinstance(x, tf.Tensor):
#         return x.numpy().decode()
#     else:
#         return x
    
# def ogg_fn(filename):
#     return os.path.join(constants.base_dir, 'train_audio', tensor_string(filename))

def duration(x):
    return librosa.get_duration(path=x)
    # return librosa.get_duration(path=ogg_fn(tf.constant(x)))

# def load_audio_tensor(filename, primary_label):
#     # https://www.tensorflow.org/tutorials/audio/transfer_learning_audio
#     # filename_ogg = tf.py_function(ogg_fn, [filename], tf.string)
#     # x = tf.io.read_file(filename_ogg)
    
#     # x = tf.io.read_file(filename)
#     # x = tfio.audio.decode_vorbis(x)
#     # x = x[:, 0]

#     audio_tensor = tfio.audio.AudioIOTensor(filename, dtype='float32')
#     x = audio_tensor.to_tensor()[:,0]
#     # eps = tf.reduce_max(tf.abs(x))*1e-5
#     # audio_start, audio_stop = tfio.audio.trim(x, epsilon=eps, axis=0)
#     # x = x[audio_start:audio_stop]


#     # # x = x - tf.reduce_mean(x)  # TO DO:  uncomment so that DC component is removed
#     x, _ = clean_audio(x)
#     return x, primary_label

def clean_audio(audio, primary_label=-1):
    largest_val = tf.reduce_max(tf.abs(audio))
    silence_thresh = largest_val * 1e-5
    non_silence = tf.where(tf.abs(audio) > silence_thresh)
    first = non_silence[0][0]
    last = non_silence[-1][0]
    audio = audio[first:last+1]
    # x = x - tf.reduce_mean(x)  # TO DO:  uncomment so that DC component is removed

    audio = audio / largest_val
    return audio, primary_label

def shift_audio(audio, label):
    shift = tf.random.uniform((), 0, len(audio), dtype=tf.int32)
    # shift = tf.py_function(lambda x: random.randint(0, x), [len(audio)], tf.int32)
    audio_shifted = tf.concat([audio[shift:], audio[:shift]], axis=0)    
    return audio_shifted, label

def chop_audio(audio, label):
    """Remove a random amount of data from the beginning of the audio

    This relates to Srengel et al's approach by allowing for different start points for a clip.

    Args:
        audio: input
        label: class id
    """
    if len(audio)>constants.frame_length:
        max_chop = len(audio)-constants.frame_length
        # NOTE: there appears to be a seed that is reset for each iteration through a dataset,
        # Making the same amount (sometimes) chopped from an audio clip
        chop_samples = tf.random.uniform((), 0, max_chop+1, dtype=tf.int32)
        return audio[chop_samples:], label
    else:
        return audio, label

def match_lengths(orig_audio, other_audio):
    len_orig = len(orig_audio)
    if len(other_audio) < len_orig:
        num_copies = tf.math.ceil(len_orig / len(other_audio))
        num_copies = tf.cast(num_copies, tf.int32)
        other_audio = tf.tile(other_audio, (num_copies,))
    other_audio = other_audio[:len_orig]
    return other_audio

def combine_recordings(orig_audio, label, index_filename_list, weight=None):
    """Add recordings with the same label

    (See Srengel et al., "Audio based bird species identification using deep learning techniques",
     CLEF2016 Working Notes,
     https://ceur-ws.org/Vol-1609/16090547.pdf)

    Args:
        orig_audio: Audio to augment
        label: class id
        index_filename_list: list of filename lists (a list for each possible class id)
        weight: weight to apply to orig_audio [Default: random value in [.5, 1.]
    """
    len_orig = len(orig_audio)
    
    other_file = tf.py_function(lambda x: random.choices(index_filename_list[x]),
                              [label],
                              tf.string)
    other_audio = load_audio_augment(other_file)
    other_audio = match_lengths(orig_audio=orig_audio, other_audio=other_audio)
    
    if weight is None:
        weight = tf.random.uniform((), .5, 1.)
    # weight = tf.py_function(lambda x: random.uniform(x, 1.), [tf.constant(.5)], tf.float32)
    # augment_audio = weight * orig_audio + (1-weight) * other_audio
    augment_audio = tf.reduce_sum([weight*orig_audio, (1-weight)*other_audio],axis=0)

    return augment_audio, label

def frame_audio(audio, label):
    # num_samples = len(audio)
    # num_repeats = tf.py_function(math.ceil, [constants.fixed_num_samples/num_samples], tf.int32)
    # if num_repeats > 1:
    #     audio = tf.tile(audio, (num_repeats,) )
    # if len(audio)>constants.fixed_num_samples:
    #     audio = audio[:constants.fixed_num_samples]
    audio = tf.signal.frame(audio, frame_length=constants.frame_length, frame_step=constants.frame_step, pad_end=True)
    return audio[:constants.fixed_num_frames], label
