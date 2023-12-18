from keras.src.engine import data_adapter
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from . import best_spec, fftlayer
from birdtraining import metrics

import constants

model2_input_shape = (299, 299, 3)
def get_mobilenet(num_classes=264):
    m = tf.keras.applications.MobileNetV3Small(input_shape=model2_input_shape,
                                               include_top=False,
                                               classes=num_classes,
                                               weights='imagenet',
                                               include_preprocessing=False)
    return m

def get_mobilenet_variable_layers():
    m = get_mobilenet()
    variable_layers = [i for i, l in enumerate(m.layers) if len(l.variables)>0]
    return variable_layers

def get_inception(num_classes=264, regularizer=None):
    m = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', pooling='avg', include_top=False)
    preds = tf.keras.layers.Dense(num_classes, kernel_regularizer=regularizer, name=f'predictions{num_classes}')
    return m, preds


class BestFrameModel(tf.keras.Model):
    """Wrapper to Keras model to use Weak Labels during training and the highest score during testing

        Based on [Deep CNN framework for audio event recognition using weakly labeled web data](https://deepai.org/publication/deep-cnn-framework-for-audio-event-recognition-using-weakly-labeled-web-data)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, data):
        # https://github.com/keras-team/keras/blob/v2.13.1/keras/engine/training.py#L1077
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # best = best_spec.BestSpec()
        # xxx = tf.map_fn(lambda xx: best(xx)[0], x, fn_output_signature=tf.float32)
        xxx = tf.map_fn(lambda xx: self._best_frame_pred(xx), data, fn_output_signature=tf.float32)
        with tf.GradientTape() as tape:
            y_pred = self(xxx, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred, sample_weight=sample_weight)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    def test_step(self, data):
        x, y = data
        y = y[tf.newaxis]
        y_pred = self.model(x, training=False)
        y_pred = tf.reduce_max(y_pred, axis=0)
        y_pred = y_pred[tf.newaxis]
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    def _best_frame_pred(self, data):
        x, y = data[:2]
        not_silence = tf.reduce_max( tf.abs(x), axis=1 ) > 0
        x_infer = x[not_silence]
        y_pred = self(x_infer, training=False)
        best = tf.argmax(y_pred[:, y])
        return x_infer[best]

def get_feats_norm_layers(mean, variance, drop_rate):
    """Return Preprocessing layers: FFT, Normalization, Dropout

       Args:
           mean: provided mean FFT mean
           variance: provided mean FFT variance
           drop_rate: rate to drop FFT values during training
    """
    fft = fftlayer.FFT(name='spectrogram', drop_rate=None)
    # best = best_spec.BestSpec()
    norm = layers.Normalization(axis=None, mean=mean, variance=variance)
    dropout = layers.Dropout(drop_rate)
    # return fft, best, norm
    return fft, norm, dropout

def get_feats_norm_stats(ds):
    """Calculate mean and variance of FFT values of datasets

       Old method. Using layers.Normalization instead.

       Args:
           ds: dataset for computing noramlization values
    """
    fft, norm = get_feats_norm_layers(None, None)
    fft_ds = (fft(x) for x in ds)
    norm.adapt(fft_ds)
    return norm.mean.numpy().item(), norm.variance.numpy().item()
    

# https://www.tensorflow.org/tutorials/images/transfer_learning
class SpectrogramModel(BestFrameModel):
    """Spectrogram-based modeling using InceptionV3

    (See Sevilla and Glotin, "Audio bird classification with inception-v4 extended with time and time-frequency attention mechanisms",
     LifeCLEF 2017 Working Notes,
     http://ceur-ws.org/Vol-1866/paper_177.pdf)

    Args:
        num_classes (int): Number of output classes
        trainable_start (int): Layer number of Inception V3 to start adaptive training
        l2 (float): Weight for L2 regularization
        mean_variance (tuple): Mean and Variance for normalizing spectrograms
    """
    def __init__(self, num_classes, trainable_start=None, l2=1e-4, mean_variance=None, drop_rate=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean, self.variance = (None, None) if mean_variance is None else mean_variance
        kernel_regularizer = tf.keras.regularizers.L2(l2=l2) if l2 else None
        self.model = tf.keras.Sequential([
            layers.Input(shape=(constants.frame_length,)),
            *get_feats_norm_layers(mean=self.mean, variance=self.variance, drop_rate=drop_rate),
            *get_inception(num_classes, kernel_regularizer),
        ])
        if trainable_start:
            for layer in self.model.layers[0].get_layer('inception_v3').layers[:trainable_start]:
                layer.trainable=False
        
    def call(self, inputs):
        outputs = self.model(inputs)
        return outputs


# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/tutorials/keras/keras_tuner
def setup_model(num_classes, mean_variance=None, hp=None, learning_rate=3e-4, trainable_start=-1, l2=1e-4, drop_rate=.1):
    """Setup model with hyperparameters

    Args:
        num_classes (int): Number of output classes

    Returns:
        Compiled SpectrogramModel
    """

    learning_rate = learning_rate if hp is None else hp.Float('learning_rate', min_value=1e-6, max_value=1e-2, sampling='log')
    drop_rate = drop_rate if hp is None else hp.Float('drop_rate', min_value=0., max_value=.5, step=.05)
    trainable_start = None #trainable_start if hp is None else hp.Choice('trainable_start', get_mobilenet_variable_layers())
    l2 = None #l2 if hp is None else hp.Float('l2_regularizer', min_value=1e-5, max_value=1e-1, step=2, sampling='log')
    model = SpectrogramModel(num_classes, trainable_start=trainable_start, l2=l2, mean_variance=mean_variance, drop_rate=drop_rate)
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),],
                 run_eagerly=None)
    return model
