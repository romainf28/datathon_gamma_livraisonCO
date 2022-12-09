"""Deprecated: only used for load control project."""
import tensorflow as tf


def mae_ts(y_true, y_pred):
    """Not used in ecospace. computed MAE on lsat timestamps of windows instead of 
    the whole window.
    Allow fair comparison between LSTM anc causal convolution models."""
    return tf.keras.losses.mean_absolute_error(y_true[:, -1, :], y_pred[:, -1, :])


def mae_perc_ts(y_true, y_pred):
    """Not used in ecospace. computed MAE on lsat timestamps of windows instead of 
    the whole window.
    Allow fair comparison between LSTM anc causal convolution models."""
    return tf.keras.metrics.mean_absolute_percentage_error(y_true[:, -1, :], y_pred[:, -1, :])
