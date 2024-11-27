import tensorflow as tf
from keras.src.utils import io_utils

class EarlyStoppingPoorStart(tf.keras.callbacks.EarlyStopping):
    # https://github.com/tensorflow/tensorflow/blob/v2.13.1/tensorflow/python/keras/callbacks.py#L1812
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if not self._is_improvement(current, self.baseline):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            io_utils.print_msg(f"Stopping early CURRENT {current} BASELINE {self.baseline}")


# class ReduceLROnPlateauMod(tf.keras.callbacks.ReduceLROnPlateau):
#     def __init__(self, **kwargs):
#         super().__init__(kwargs)
#         self.best_weights = None
        
#     def on_epoch_end(self, epoch, logs=None):
#         old_lr = self.model.optimizer.learning_rate
#         print("EPOCH",epoch)
#         print("LOGS",logs)
#         super().on_epoch_end(epoch, logs)
#         new_lr = self.model.optimizer.learning_rate
#         if old_lr == new_lr:
#             if self.wait == 0:
#                 if self.verbose > 0:
#                     io_utils.print_msg(f"Saving weights for {old_lr}")
#                 self.best_weights = self.model.get_weights()
#             else:
#                 if self.verbose > 0:
#                     io_utils.print_msg(f"In waiting period for {old_lr}")
#         else:
#             if self.verbose > 0:
#                 io_utils.print_msg(f"Plateaued. Restoring weights for {old_lr}")
#             self.model.set_weights(self.best_weights)
