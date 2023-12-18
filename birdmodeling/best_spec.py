import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp

# https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing#privileged_training_argument_in_the_call_method
class BestSpec(layers.Layer):
    """Retain "best" frame of a sequence

    Best is defined by a heuristic that looks at which frame has frequencies and times that jointly
    have a lot of energy.

    It has some similarity to the Median Clipping performed in:
    Lasseck, "Bird song classification in field recordings: winning solution for nips4b 2013 competition",
    https://www.tierstimmenarchiv.de/RefSys/Nips4b2013NotesAndSourceCode/WorkingNotes_Mario.pdf

    Args:
        percentile (int): which percentile to define as a high threshold
    """
    def __init__(self, percentile=50, **kwargs):
        super().__init__(**kwargs)
        self.percentile = percentile
    def call(self, inputs, training=False):
        if training:
            # return tf.map_fn(self._get_best, inputs, fn_output_signature=tf.float32)
            res = self._get_best(inputs)
            return res
        else:
            return inputs
    def _get_best(self, x):
        thresh_rows = 2+tfp.stats.percentile(x, self.percentile, axis=1, keepdims=True)
        thresh_cols = 2+tfp.stats.percentile(x, self.percentile, axis=2, keepdims=True)
        strong_rows = x>thresh_rows
        strong_cols = x>thresh_cols
        strong = tf.cast(strong_rows & strong_cols, tf.int64)
        strong = tf.reduce_sum(strong, axis=[1, 2, 3])
        order = tf.argsort(strong, direction='DESCENDING')
        best = order[0]
        return x[best]
