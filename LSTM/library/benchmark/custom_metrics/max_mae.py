"""This file contains useful custom metrics for Ecospace project. These functions are compile thanks to the tf function decorator. Graph representation allows fast compute."""
from keras import backend

import tensorflow as tf
import pickle

def mae_error_(y_mean, y_std):
    """Give the actual MAE between the original 
    non scaled values without computing the transformation.
    Not working properly. 
    Using this function with multi Output makes no sense:
    averaging the error between different unscaled units will
    results in bad training."""
    @tf.function
    def mae_error(y_true, y_pred):
        return tf.math.reduce_mean(tf.math.multiply(
            tf.abs((y_true - y_pred)),
            y_std
        ))
    return mae_error


def mse_error_(y_mean, y_std):
    """Give the actual MSE between the original 
    non scaled values without computing the transformation.
    Not working properly. 
    Using this function with multi Output makes no sense:
    averaging the error between different unscaled units will
    results in bad training."""
    @tf.function
    def mse_error(y_true, y_pred):
        obj =  tf.math.multiply(
            tf.math.squared_difference(y_pred, y_true),
            tf.math.square(y_std)
        )
        return tf.math.reduce_mean(obj)

    return mse_error

def absolute_percentage_error_(y_mean, y_std):
    """Give the actual MAE% between the original 
    non scaled values without computing the transformation.
    Not working properly. I gave the formula in the 
    ecospace report section 5. May have issue with bad reduce.
    """
    @tf.function
    def absolute_percentage_error(y_true, y_pred):

        num = tf.math.multiply(
            tf.abs((y_true - y_pred)), 
            y_std
        )
        denum= tf.math.add(
            tf.math.multiply(y_true,y_std),
            y_mean
        )

        result = tf.math.divide(
            num,
            tf.math.maximum(
                tf.abs(y_true),
                tf.keras.backend.epsilon()
            )
        )

        return 100.* tf.math.reduce_mean(result)
    return absolute_percentage_error


def max_absolute_percentage_error_(y_mean, y_std):
    """Not working properly. I gave the formula in the 
    ecospace report section 5. May have issue with bad reduce.
    
    Give the actual max MAE% between the original 
    non scaled values without computing the transformation.
    The max is between the output. This could result in a sequential
    train of a common network to several outputs. This methods failed.
    """
    @tf.function
    def max_absolute_percentage_error(y_true, y_pred):

        num = tf.math.multiply(
            tf.abs((y_true - y_pred)), 
            y_std
        )
        denum= tf.math.add(
            tf.math.multiply(y_true,y_std),
            y_mean
        )

        result = tf.math.divide(
            num,
            tf.math.maximum(
                tf.abs(y_true),
                tf.keras.backend.epsilon()
            )
        )

        return 100.* tf.math.reduce_max(result, axis=-1)
    return max_absolute_percentage_error

@tf.function
def max_mean_absolute_error(y_true, y_pred):
    """Stupid function. Not working as intended and not used."""
    diff = tf.abs((y_true - y_pred))
    return tf.math.reduce_max(diff, axis=-1)