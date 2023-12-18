class SetupModel:
    """Wrap a model's setup function for use in hyperparameter tuning

    Args:
        num_classes: Number of model outputs
        mean_variance (tuple): Mean and variance of spectrogram values
    """
    def __init__(self, num_classes, mean_variance, func):
        self.num_classes = num_classes
        self.mean_variance = mean_variance
        self.func = func
    def setup_model_wrap(self, hp=None):
        """Wrap setup function

        Arg:
            hp: hyperparameters (default: None)

        Returns:
            model's setup function
        """
        return self.func(self.num_classes, self.mean_variance, hp)
