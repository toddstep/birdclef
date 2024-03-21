import os
import pickle

import numpy as np

os.environ['BASE_DIR'] = '.'

from birdmodeling import spectrogram_model
import constants

# https://stackoverflow.com/questions/15233340/getting-rid-of-n-when-using-readlines
with open('competition_classes.txt', 'r') as f:
    competition_classes = f.read().splitlines()
num_classes = len(competition_classes)

audio_stats_file = "audio_stats.pickle"
audio_stats = pickle.load(open(audio_stats_file, 'rb') )
print(audio_stats)
spec_model = spectrogram_model.setup_model(num_classes, mean_variance=audio_stats)
spec_model.load_weights('spectrogram_model_TUNE').expect_partial()


#https://www.tensorflow.org/guide/keras/serialization_and_saving 
spec_model(np.random.random((1, constants.frame_length)));
spec_model.export('./export_model')