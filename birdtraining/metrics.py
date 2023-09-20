import tensorflow as tf

# https://deepai.org/publication/deep-cnn-framework-for-audio-event-recognition-using-weakly-labeled-web-data
class WeakLoss(tf.keras.losses.SparseCategoricalCrossentropy):
    """Weak loss for sequence data

    The sequence element with the highest score for the ground truth is selected.
    Categorical cross-entropy loss is computed only for that element
    """
    # https://www.tensorflow.org/tutorials/quickstart/beginner
    def __init__(self):
        super().__init__(from_logits=True)
    
    # https://stackoverflow.com/questions/61919774/unexpected-keyword-argument-sample-weight-when-sub-classing-tensor-flow-loss-c
    def call(self, y_true, y_pred):
        # https://stackoverflow.com/questions/44667395/tensorflow-how-to-apply-a-function-to-the-last-dimension-of-a-tensor
        y_pred_max = tf.map_fn(self._get_max, [y_true, y_pred], tf.float32)
        ret_loss = super().call(y_true, y_pred_max)
        return ret_loss

    def _get_max(self, gt_recording):
        gt, recording = gt_recording
        best_gt_frame = tf.argmax(recording[:, gt])
        max_frame = recording[best_gt_frame]
        return max_frame

def whole_recording_predict(y_pred):
    """Compute which class has max score in a recording

    Arg:
        y_pred: Predictions for recording sequence

    Returns:
        Predicted class for sequence
    """
    # https://stackoverflow.com/questions/50423129/how-do-i-zip-tensors-in-tensorflow-when-the-dimensions-dont-match
    max_of_classes_per_recording = tf.reduce_max(y_pred, axis=1)
    argmax_per_recording = tf.argmax(max_of_classes_per_recording, axis=1, output_type=tf.int32)
    # argmax_per_recording = argmax_per_recording[:, tf.newaxis]
    return argmax_per_recording

def whole_recording_accuracy(y_true, y_pred):
    """Compute if prediction is correct for a recording

    Args:
        y_true: Ground truth for recording
        y_pred: Predictions for recording sequence

    Returns:
        Accuracy:  1 if correct or 0 if incorrect.
    """
    y_true = tf.cast(y_true, tf.int32)
    whole_predict = whole_recording_predict(y_pred)
    is_correct = whole_predict == y_true
    acc = tf.cast(is_correct, tf.float32)
    return acc