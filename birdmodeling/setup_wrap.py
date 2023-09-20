class SetupModel:
    """Wrap a model's setup function for use in hyperparameter tuning
    """
    def __init__(self, num_classes, func):
        self.num_classes = num_classes
        self.func = func
    def setup_model_wrap(self, hp=None):
        """Wrap setup function

        Arg:
            hp: hyperparameters (default: None)

        Returns:
            model's setup function
        """
        return self.func(self.num_classes, hp)
