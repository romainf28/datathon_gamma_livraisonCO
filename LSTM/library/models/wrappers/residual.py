"""This file contrain the residual wrapper, not usable when predicting several timestamps ahead. Not used in ecospace project."""
import tensorflow as tf


class ResidualWrapper(tf.keras.Model):
    """Wrap any model generator with an instance of this class
    to predict the change in outputs instead of the actual value.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each time step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta
