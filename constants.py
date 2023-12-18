import os

model_sr = 22050
frame_length = 5*model_sr
frame_step = 5*model_sr
fixed_num_frames = 12
fixed_num_samples = frame_length+(fixed_num_frames-1)*frame_step
base_dir = os.environ['BASE_DIR']
cached_dir = "../birdclef-cache"
tune_outdir = "../birdclef-tune"
is2020data = base_dir.endswith('birdsong-recognition')
metadata_csv = "train.csv" if is2020data else "train_metadata.csv"