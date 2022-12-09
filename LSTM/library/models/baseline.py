"""This file contains baselines models. They makes up an excellent comparison point with timeseries models."""
from typing import List

import tensorflow as tf


class Baseline(tf.keras.Model):
    """USed in load control project only. Predict the next points to 
    be the same as the last one at each timestamp."""
    
    def __init__(self, label_index: int = 1):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


class MultiStepLastBaseline(tf.keras.Model):
    """Predict the next points to be the same as the last given one."""
    def __init__(self, OUTSTEPS: int):
        super().__init__()
        self.OUTSTEPS = OUTSTEPS

    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, self.OUTSTEPS, 1])


class RepeatBaseline(tf.keras.Model):
    """Not coded yet. Won't be used in the project."""
    def call(self, inputs):
        return inputs
