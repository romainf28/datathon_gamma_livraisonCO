"""Old functions. Not used in ecospace."""
from ..models.base.template import Predictor
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf


def y_vectors(model: Predictor, label: str = "TT21") -> Tuple[Dict[str, np.array]]:
    """Used in Load control project to definethe y vectors from model as Predictor 
    instance."""
    y_true = {skid: list() for skid in model.window.sites}
    y_pred = {skid: list() for skid in model.window.sites}
    for skid_index, skid in enumerate(model.window.sites):
        # windows.test: List(Tuple Features, labels): skid
        for X, y in model.window.test[skid_index]:
            y_pred[skid].append(model.models[skid].predict(X)[:, -1, :].reshape(-1))
            y_true[skid].append(tf.reshape(y[:, -1, :], -1, name=None).numpy())

        # normalize
        y_pred[skid] = (
                np.concatenate(y_pred[skid], axis=0) * model.window.train_std.loc[skid][label]
                + model.window.train_mean.loc[skid][label]
        )
        y_true[skid] = (
                np.concatenate(y_true[skid], axis=0) * model.window.train_std.loc[skid][label]
                + model.window.train_mean.loc[skid][label]
        )

    return y_true, y_pred
