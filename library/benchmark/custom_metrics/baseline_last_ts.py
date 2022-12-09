"""Deprecated: only used for load control project."""
import tensorflow as tf

alpha = 1e-3


def bsln_ts(y_true, y_pred):
    """Not used in ecospace. computed MAE on lsat timestamps of windows instead of 
    the whole window.
    Allow fair comparison between LSTM anc causal convolution models."""
    # baseline keeps same output
    baseline = tf.keras.losses.mean_absolute_error(y_true[:, -1, :], y_true[:, -2, :])
    m_ = tf.keras.losses.mean_absolute_error(y_true[:, -1, :], y_pred[:, -1, :])

    # this is a score. the higher the better. Additive smoothing
    return (alpha + m_) / (alpha + baseline)
