"""Deprecated: only used for load control project."""
import tensorflow as tf


def bsln_ts(y_true, y_pred):
    """Compute r2 score."""
    # baseline keeps same output
    baseline = tf.keras.losses.mean_squared_error(y_true[:, -1, :], y_true[:, -2, :])
    m_ = tf.keras.losses.mean_squared_error(y_true[:, -1, :], y_pred[:, -1, :])

    return 1 - m_ / baseline
