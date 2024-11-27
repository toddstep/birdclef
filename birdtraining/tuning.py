import os

import keras_tuner as kt
import tensorflow as tf

import constants
from birdmodeling import setup_wrap

num_parallel = tf.data.AUTOTUNE

# https://www.tensorflow.org/tutorials/keras/keras_tuner
def build_model_hyperparameters(setup, num_classes, mean_variance, datasets, class_weight, callbacks, batch_size):
    """
    Setup model with hyperparameter tuning

    Args:
        setup (function): model setup
        num_classes (int): Number of output classes
        mean_variance (Tuple): mean and standard deviation of spectrograms
        datasets (dict): train and valid[ation] datasets
        class_weight (dict): Weights for loss computation when fitting model
        classbacks (list): Callback methods when fitting model
        batch_size (int): Number of recordings in batch
    """
    tuner_setup = setup_wrap.SetupModel(num_classes, mean_variance, setup)
    tuner = kt.Hyperband(tuner_setup.setup_model_wrap,
                          objective=kt.Objective("val_sparse_categorical_accuracy", direction="max"),
                          max_epochs=60,
                          factor=3,
                          hyperband_iterations=1,
                          directory=os.path.join(constants.tune_outdir,'spec_model'),
                          project_name='birdclef')
    # tuner = kt.BayesianOptimization(tuner_setup.setup_model_wrap,
    #                       objective=kt.Objective("val_sparse_categorical_accuracy", direction="max"),
    #                       max_trials=10,
    #                       directory=os.path.join(constants.tune_outdir,'spec_model'),
    #                       project_name='birdclef2023')

    tuner.search(datasets['train'].padded_batch(batch_size).prefetch(num_parallel),
                 validation_data=datasets['valid'].prefetch(num_parallel),
                 epochs=50,
                 callbacks=callbacks,
                 class_weight=class_weight)

    tuner.results_summary()
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    return best_model
