"""This file contain the plot_training function."""
from ..models.base.template import Predictor
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_training(model: Predictor, stat='loss') -> None:
    """This function is used in notebooks to monitor the training curve.
    Useful to set earlyStopping parameters or to compare optimizers, learning rates...
    
    It will compare the value of the given statistic with the one on valid dataset."""
    # plt.title("Model Loss on validation data - all points [best weights restored]")
    lim_inf, lim_sup = np.inf, 0

    plt.plot(model.history.history[stat], label='train')
    plt.plot(model.history.history["val_{}".format(stat)], label='valid')
    minimum = min(
        model.history.history["val_{}".format(stat)]
    )

    lim_inf, lim_sup = min(minimum, lim_inf), max(minimum, lim_sup)

    plt.hlines(lim_inf, *plt.xlim(), linestyles="dotted")
    plt.hlines(lim_sup, *plt.xlim(), linestyles="dotted")
    plt.title(str(lim_inf)[:3] + "-" + str(lim_sup)[:3])

    plt.xlabel('epoch')
    plt.ylabel(stat)
    plt.legend()

    plt.show()
