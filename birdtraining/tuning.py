import os

import keras_tuner as kt

import constants
from birdmodeling import setup_wrap

# https://www.tensorflow.org/tutorials/keras/keras_tuner
def build_model_hyperparameters(setup, num_classes, datasets, class_weight, callbacks, batch_size):
    """
    Setup model with hyperparameter tuning

    Args:
        setup (function): model setup
        num_classes (int): Number of output classes
        datasets (dict): train and valid[ation] datasets
        class_weight (dict): Weights for loss computation when fitting model
        classbacks (list): Callback methods when fitting model
        batch_size (int): Number of recordings in batch
    """
    tuner_setup = setup_wrap.SetupModel(num_classes, setup)
    # tuner = kt.Hyperband(tuner_setup.setup_model_wrap,
    #                       objective=kt.Objective("val_whole_recording_accuracy", direction="max"),
    #                       max_epochs=16,
    #                       factor=2,
    #                       directory=os.path.join(constants.tune_outdir,'spec_model'),
    #                       project_name='birdclef2023')
    tuner = kt.BayesianOptimization(tuner_setup.setup_model_wrap,
                          objective=kt.Objective("val_whole_recording_accuracy", direction="max"),
                          max_trials=20,
                          directory=os.path.join(constants.tune_outdir,'spec_model'),
                          project_name='birdclef2023')

    tuner.search(datasets['train'].batch(batch_size).prefetch(4),
                 validation_data=datasets['valid'].batch(batch_size).prefetch(4),
                 epochs=10,
                 callbacks=callbacks,
                 class_weight=class_weight)

    tuner.results_summary()
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    return best_model