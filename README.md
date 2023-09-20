# birdclef

Train a classifier for the [Cornell Birdcall Identification](https://www.kaggle.com/competitions/birdsong-recognition) data.
The training uses a pretrained [MobileNet](https://doi.org/10.48550/arXiv.1704.04861) neural network.

The training loss function incorporates the following:
- Recording-level loss, based on [Deep CNN framework for audio event recognition using weakly labeled web data](https://deepai.org/publication/deep-cnn-framework-for-audio-event-recognition-using-weakly-labeled-web-data)
- Weights using class frequencies in the training set

The validation set incorporates the following:
- Use of only high-frequency classes. Classes with less than 25 samples are excluded from the validation fold. The intent is to reserve data from rare classes for training -- even though there will be no samples from such classes for validation.
- Balanced data selection. For each (high-frequency) class, choose 5 validation samples.

Information on preprocessing and training is demonstrated in [birdclef-modeling.ipynb](birdclef-modeling.ipynb).
Note that the current training algorithm appears to be severely overfitting the model. That is, it has an accuracy of 72% on the training set and 35% on the validation set.
