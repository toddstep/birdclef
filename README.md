# birdclef

Train a classifier for the [Cornell Birdcall Identification](https://www.kaggle.com/competitions/birdsong-recognition) data.
It uses a pretrained [Inception](https://openaccess.thecvf.com/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf) neural network.

The training incorporates a recording-level loss, based on [Deep CNN framework for audio event recognition using weakly labeled web data](https://deepai.org/publication/deep-cnn-framework-for-audio-event-recognition-using-weakly-labeled-web-data)

Information on the data and training is demonstrated in [birdclef-modeling.ipynb](birdclef-modeling.ipynb).
