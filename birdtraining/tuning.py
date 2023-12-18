import os

import keras_tuner as kt

import constants
from birdmodeling import setup_wrap

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
                          max_epochs=16,
                          factor=3,
                          directory=os.path.join(constants.tune_outdir,'spec_model'),
                          project_name='birdclef')
    # tuner = kt.BayesianOptimization(tuner_setup.setup_model_wrap,
    #                       objective=kt.Objective("val_whole_recording_accuracy", direction="max"),
    #                       max_trials=20,
    #                       directory=os.path.join(constants.tune_outdir,'spec_model'),
    #                       project_name='birdclef2023')

    tuner.search(datasets['train'].padded_batch(batch_size).prefetch(2),
                 validation_data=datasets['valid'].prefetch(2),
                 epochs=6,
                 callbacks=callbacks,
                 class_weight=class_weight)

    tuner.results_summary()
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    return best_model