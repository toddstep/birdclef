from keras.src.engine import data_adapter
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from . import fftlayer
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

def get_inception(num_classes=264, weights=None, feat_drop_rate=None, regularizer=None):
    print("WEIGHTS", weights)
    m = tf.keras.applications.inception_v3.InceptionV3(weights=weights, pooling='avg', include_top=False)
    preds = tf.keras.layers.Dense(num_classes, kernel_regularizer=regularizer, name=f'predictions{num_classes}')
    if feat_drop_rate is None:
        return m, preds
    else:
        drop = tf.keras.layers.Dropout(feat_drop_rate)
        return m, drop, preds


class BestFrameModel(tf.keras.Model):
    """Wrapper to Keras model to use Weak Labels during training and the highest score during testing

        Based on [Deep CNN framework for audio event recognition using weakly labeled web data](https://deepai.org/publication/deep-cnn-framework-for-audio-event-recognition-using-weakly-labeled-web-data)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, data):
        # https://github.com/keras-team/keras/blob/v2.13.1/keras/engine/training.py#L1077
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        xxx = tf.map_fn(lambda xx: self._best_frame_pred(xx), data, fn_output_signature=tf.float32)
        return super().train_step((xxx, y, sample_weight))
        # with tf.GradientTape() as tape:
        #     y_pred = self(xxx, training=True)
        #     loss = self.compute_loss(y=y, y_pred=y_pred, sample_weight=sample_weight)
        # self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        # self.compiled_metrics.update_state(y, y_pred)
        # return {m.name: m.result() for m in self.metrics}
    def test_step(self, data):
        x, y = data
        y = y[tf.newaxis]
        y_pred = self.model(x, training=False)
        y_pred = tf.reduce_max(y_pred, axis=0)
        y_pred = y_pred[tf.newaxis]
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    @tf.function
    def _best_frame_pred(self, data):
        x, y = data[:2]
        not_silence = tf.reduce_max( tf.abs(x), axis=1 ) > 0
        x_infer = x[not_silence]
        y_pred = self(x_infer, training=False)
        best = tf.argmax(y_pred[:, y])
        return x_infer[best]

def get_feats_norm_layers(mean, variance, freq_drop_rate, freq_mask_param, time_mask_param, gauss_noise_param, reduce_db_param, do_freq_fade, do_time_fade):
    """Return Preprocessing layers: FFT, Normalization, Dropout

       Args:
           mean: provided mean FFT mean
           variance: provided mean FFT variance
           freq_drop_rate: rate to drop FFT values during training
    """
    fft = fftlayer.FFT(name='spectrogram', freq_drop_rate=freq_drop_rate, freq_mask_param=freq_mask_param, time_mask_param=time_mask_param, gauss_noise_param=gauss_noise_param, reduce_db_param=reduce_db_param, do_freq_fade=do_freq_fade, do_time_fade=do_time_fade)
    # norm = layers.Normalization(axis=None, mean=mean, variance=variance)
    # return fft, norm
    return fft,

def get_feats_norm_stats(ds):
    """Calculate mean and variance of FFT values of datasets

       Old method. Using layers.Normalization instead.

       Args:
           ds: dataset for computing noramlization values
    """
    fft, norm = get_feats_norm_layers(*([None]*8))
    fft_ds = ds.map(fft)
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
        weights (string):
        l2 (float): Weight for L2 regularization
        mean_variance (tuple): Mean and Variance for normalizing spectrograms
    """
    def __init__(self, num_classes, weights=None, trainable_start=None, l2=1e-4, mean_variance=None, feat_drop_rate=None, freq_drop_rate=None, freq_mask_param=None, time_mask_param=None, gauss_noise_param=None, reduce_db_param=None, do_freq_fade=False, do_time_fade=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean, self.variance = (None, None) if mean_variance is None else mean_variance
        kernel_regularizer = tf.keras.regularizers.L2(l2=l2) if l2 else None
        self.model = tf.keras.Sequential([
            layers.Input(shape=(constants.frame_length,)),
            *get_feats_norm_layers(mean=self.mean,
                                   variance=self.variance,
                                   freq_drop_rate=freq_drop_rate,
                                   freq_mask_param=freq_mask_param,
                                   time_mask_param=time_mask_param,
                                   gauss_noise_param=gauss_noise_param,
                                   reduce_db_param=reduce_db_param,
                                   do_freq_fade=do_freq_fade,
                                   do_time_fade=do_time_fade,),
            *get_inception(num_classes, weights=weights, feat_drop_rate=feat_drop_rate, regularizer=kernel_regularizer),
        ])
        if trainable_start:
            for layer in self.model.layers[0].get_layer('inception_v3').layers[:trainable_start]:
                layer.trainable=False
        
    def call(self, inputs):
        outputs = self.model(inputs)
        return outputs


# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/tutorials/keras/keras_tuner
def setup_model(num_classes, weights=None, mean_variance=None, hp=None, learning_rate=3e-4, trainable_start=-1, l2=1e-4, feat_drop_rate=None, freq_drop_rate=None, freq_mask_param=None, time_mask_param=None, gauss_noise_param=None, reduce_db_param=None, do_freq_fade=False, do_time_fade=False):
    """Setup model with hyperparameters

    Args:
        num_classes (int): Number of output classes

    Returns:
        Compiled SpectrogramModel
    """

    learning_rate = learning_rate if hp is None else hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, step=1.1, sampling="log")
    time_mask_param = time_mask_param if hp is None else hp.Choice('time_mask_param', [150, 290,])
    freq_mask_param = freq_mask_param if hp is None else hp.Choice('freq_mask_param', [150, 290,])
    reduce_db_param = reduce_db_param if hp is None else hp.Choice('reduce_db_param', [.5, .9])

    # feat_drop_rate = learning_rate if hp is None else hp.Float('feat_drop_rate', min_value=.001, max_value=.999, step=1.1, sampling="reverse_log")
    # freq_drop_rate = freq_drop_rate if hp is None else hp.Float('freq_drop_rate', min_value=0., max_value=.99, step=.1)
    
    # CITATION for learning rate steps???
    # learning_rate = learning_rate if hp is None else hp.Choice('learning_rate', [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1,])
    feat_drop_rate = None # feat_drop_rate if hp is None else hp.Choice('feat_drop_rate', [0., .1, .25, .50, .75, .90, .95, .99, .999,])
    freq_drop_rate = None # freq_drop_rate if hp is None else hp.Choice('freq_drop_rate', [0., .1, .25, .50, .75, .90, .95, .99, .999,])
    # freq_mask_param = None # freq_mask_param if hp is None else hp.Choice('freq_mask_param', [0, 10, 30, 90, 270,])
    # time_mask_param = None # time_mask_param if hp is None else hp.Choice('time_mask_param', [0, 10, 30, 90, 270,])
    gauss_noise_param = None # gauss_noise_param if hp is None else hp.Choice('gauss_noise_param', [0., .01, .03, .1, .3, 1., 3., 10.,])
    # reduce_db_param = None # reduce_db_param if hp is None else hp.Choice('reduce_db_param', [0., .1, .3, .5, .7, .9,])
    do_freq_fade = False # do_freq_fade if hp is None else hp.Boolean('do_freq_fade')
    do_time_fade = False # do_time_fade if hp is None else hp.Boolean('do_time_fade')
    trainable_start = None #trainable_start if hp is None else hp.Choice('trainable_start', get_mobilenet_variable_layers())
    l2 = None #l2 if hp is None else hp.Float('l2_regularizer', min_value=1e-5, max_value=1e-1, step=2, sampling='log')
    model = SpectrogramModel(num_classes,
                             weights=weights,
                             trainable_start=trainable_start,
                             l2=l2,
                             mean_variance=mean_variance,
                             feat_drop_rate=feat_drop_rate,
                             freq_drop_rate=freq_drop_rate,
                             freq_mask_param=freq_mask_param,
                             time_mask_param=time_mask_param,
                             gauss_noise_param=gauss_noise_param,
                             reduce_db_param=reduce_db_param,
                             do_freq_fade=do_freq_fade,
                             do_time_fade=do_time_fade)
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),],
                 run_eagerly=None)
    return model
