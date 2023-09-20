import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from . import fftlayer
from birdtraining import metrics

import constants

model2_input_shape = (500, 200, 3)
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


# https://www.tensorflow.org/tutorials/images/transfer_learning
class SpectrogramModel(tf.keras.Model):
    """Spectrogram-based modeling using MobileNetV3Small

    Args:
        num_classes (int): Number of output classes
        trainable_start (int): Layer number if MobileNetV3Small to start adaptive training
        l2 (float): Weight for L2 regularization
    """
    def __init__(self, num_classes, trainable_start=-1, l2=1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feats = tf.keras.Sequential([            
            layers.Input(shape=(None, constants.frame_length,)),
            fftlayer.FFT(),
        ])
        kernel_regularizer = tf.keras.regularizers.L2(l2=l2) if l2 else None
        self.model = tf.keras.Sequential([
            get_mobilenet(num_classes),
            # https://github.com/keras-team/keras/blob/v2.12.0/keras/applications/mobilenet_v3.py#L362-377
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
            tf.keras.layers.Conv2D(1024,
                                   (1, 1),
                                   padding='same',
                                   activation='relu',
                                   kernel_regularizer=kernel_regularizer),
            tf.keras.layers.Dropout(.1),
            tf.keras.layers.Conv2D(num_classes,
                                   (1, 1),
                                   padding='same',
                                   activation=None,
                                   kernel_regularizer=kernel_regularizer),
            tf.keras.layers.Flatten(),
        ])
        for layer in self.model.layers[0].layers[:trainable_start]:
            layer.trainable=False
        
    def call(self, inputs):
        feats = self.feats(inputs)
        rtn = tf.map_fn(lambda x: self.model(x), feats)
        return rtn
        

# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://www.tensorflow.org/tutorials/keras/keras_tuner
def setup_model(num_classes, hp=None, learning_rate=3e-4, trainable_start=-1, l2=1e-4):
    """Setup model with hyperparameters

    Args:
        num_classes (int): Number of output classes

    Returns:
        Compiled SpectrogramModel
    """

    learning_rate = learning_rate if hp is None else hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    trainable_start = None #trainable_start if hp is None else hp.Choice('trainable_start', get_mobilenet_variable_layers())
    l2 = None #l2 if hp is None else hp.Float('l2_regularizer', min_value=1e-5, max_value=1e-1, step=2, sampling='log')
    model = SpectrogramModel(num_classes, trainable_start=trainable_start, l2=l2)
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 loss=metrics.WeakLoss(),
                 metrics=[metrics.whole_recording_accuracy],
                 run_eagerly=None)
    return model
