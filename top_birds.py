import base64
import os
import pickle
import sys

os.environ['BASE_DIR'] = '.'

import librosa
import numpy as np
import tensorflow as tf

# from birdmodeling import spectrogram_model
import constants
import preprocessing

# https://stackoverflow.com/questions/15233340/getting-rid-of-n-when-using-readlines
with open('competition_classes.txt', 'r') as f:
    competition_classes = f.read().splitlines()
num_classes = len(competition_classes)

tmp_file = '/tmp/bbb'

# audio_stats_file = "audio_stats.pickle"
score_threshold_file = 'score_threshold.pickle'
score_threshold = pickle.load(open(score_threshold_file, 'rb')) if os.path.exists(score_threshold_file) else None
# audio_stats = pickle.load(open(audio_stats_file, 'rb') )
# print(audio_stats)

# spec_model = spectrogram_model.setup_model(num_classes, mean_variance=audio_stats)
# spec_model.load_weights('spectrogram_model_TUNE').expect_partial()

spec_model = tf.saved_model.load('./export_model')

def process_file(file):
    top_scores_per_class = get_reduced_scores(file)
    birds_scores = sorted(list(zip(competition_classes, top_scores_per_class.numpy())), key=lambda x:x[1], reverse=True)
    hits = [(bird, score) for bird, score in birds_scores if score >= score_threshold]
    return hits

def get_reduced_scores(file):
    audio, _ = librosa.load(file, sr=constants.model_sr, mono=True)
    audio, _ = preprocessing.clean_audio(audio, None)
    framed_audio, _ = preprocessing.frame_audio(audio, None)
    all_scores = spec_model.serve(framed_audio)
    return tf.reduce_max(all_scores, axis=0)

def lambda_handler(event, context=None):
    with open(tmp_file, 'wb') as f:
        f.write(base64.b64decode(event['body'].encode()))
    classes_scores = process_file(tmp_file)
    # https://ellisvalentiner.com/post/serializing-numpyfloat32-json/
    top_results = [(x[0], np.float64(x[1])) for x in classes_scores if x[1] > 0]
    print(top_results)

    return {'top_results': top_results}

if __name__ == "__main__":
    file = sys.argv[1]
    classes_scores = process_file(file)
    print(classes_scores)
