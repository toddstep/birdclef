# birdclef

## Train a classifier for the [Cornell Birdcall Identification](https://www.kaggle.com/competitions/birdsong-recognition) data:
* data split of audio recordings:
    * train: 19151 recordings
    * test: 2222 recordings
* outputs for 264 bird species
* pretrained [Inception](https://openaccess.thecvf.com/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf) neural network backbone
* recording-level metric: identify the _primary_ bird in a recording
* loss based on [Deep CNN framework for audio event recognition using weakly labeled web data](https://deepai.org/publication/deep-cnn-framework-for-audio-event-recognition-using-weakly-labeled-web-data)
* training augmentation: remove a random amount of audio from the beginning of each recording
* hyperparameter tuning:
    *  learning rate (of Adam optimizer)
    *  drop rate (of input spectrogram features)

## Results:
* tuned hyperparameters:
    * learning rate 0.00021564782468849842
    * drop rate 0.05
* 40 training epochs
* recording-level accuracy: 73.7%

## Additional information is available in the [Birdclef Training Notebook](birdclef-modeling.ipynb).
