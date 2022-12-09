"""Base Class model for conv, lstm, gru. Has a builtin compile and fit functions."""
from ...benchmark.custom_metrics.baseline_last_ts import bsln_ts
from ...benchmark.custom_metrics.mae_last_ts import mae_ts, mae_perc_ts
from ...benchmark.custom_metrics.max_mae import (
    max_absolute_percentage_error_,
    absolute_percentage_error_,
    mae_error_,
    mse_error_,
)

from typing import Tuple, Dict, List, Any

import tensorflow as tf
import tensorflow_addons as tfa


class Predictor:
    """The Predictor template will train a given model generator (id est any function returning a keras instance
    on any given WindowGenerator instance. woth the given optimizer and train parameters.

    sent no_clone as keyword argument to the function to send any non keras based model (such as wrapped models)."""

    def __init__(
        self,
        name: str,
        window,
        model_generator: Any,
        optimizer: tf.optimizers = tf.optimizers.Adam(),
        loss_mod: str = "mae",
        callback_mod: str = "EarlyStopping",
        MAX_EPOCHS: int = 400,
        wrapper=None,
        watch: str = "val_loss",
        verbose: str = "show",
        *args,
        **kwargs,
    ):
        """Constructor of the class."""
        self.name = name
        self.watch = watch
        self.window = window

        self.verbose = verbose

        # obsevation: mean delta for mae: 1e-3

        if (
            isinstance(model_generator(), tf.keras.Model)
            and "no_clone" not in kwargs.keys()
        ):
            self.model = (
                wrapper(tf.keras.models.clone_model(model_generator()))
                if wrapper
                else tf.keras.models.clone_model(model_generator())
            )
        else:
            self.model = wrapper(model_generator()) if wrapper else model_generator()

        self.optimizer = optimizer if "no_opti" not in kwargs.keys() else None
        # Compile the tensorflow model with Adam optimizer.
        # Use MAE and MAE% metrics and optimize the MSE.

        df_train_mean, df_train_std = window.df_mean, window.df_std

        y_mean = tf.constant(df_train_mean[window.label_columns].tolist())
        y_std = tf.constant(df_train_mean[window.label_columns].tolist())
        max_absolute_percentage_error = max_absolute_percentage_error_(
            y_mean=y_mean, y_std=y_std
        )

        mean_absolute_percentage_error = absolute_percentage_error_(
            y_mean=y_mean, y_std=y_std
        )

        mae_error_scaled = mae_error_(y_mean=y_mean, y_std=y_std)

        mse_error_scaled = mse_error_(y_mean=y_mean, y_std=y_std)

        metrics = [
            # tf.metrics.MeanAbsoluteError(),
            # tf.keras.metrics.MeanAbsolutePercentageError(),
            tf.keras.metrics.RootMeanSquaredError(),
            #mae_ts,
            #mae_error_scaled,
            #mean_absolute_percentage_error,
        ]
        if loss_mod == "mae":
            loss = tf.losses.MeanAbsoluteError()
            self.min_ = 3e-3
        elif loss_mod == "mse_scaled":
            loss = mse_error_scaled
            self.min_ = 1e-1
        elif loss_mod == "mse":
            loss = tf.losses.MeanSquaredError()
            self.min_ = 1e-5
        elif loss_mod == "percent_max":
            loss = max_absolute_percentage_error
            self.min_ = 5e-1
        elif loss_mod == "percent":
            loss = tf.keras.losses.MeanAbsolutePercentageError()
            self.min_ = 5e-1

        self.model.compile(
            loss=loss,
            optimizer=self.optimizer if self.optimizer else None,
            metrics=metrics,
        )

        if "no_fit" not in kwargs.keys():
            self.history = self.fit_model(MAX_EPOCHS=MAX_EPOCHS)

    def fit_model(
        self, patience: int = 70, MAX_EPOCHS: int = 60, min_delta: float = 1e-4
    ) -> tf.keras.callbacks.History():
        """Call to fit the model to the data. window is
        an attribute pointing to the given data in constructor.
        You can modify here the patience for earlystopping callback."""

        callback_ = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",  # self.watch,
            patience=patience,
            min_delta=min_delta,
            mode="min",
            restore_best_weights=True,
        )

        progress_bar = tfa.callbacks.TQDMProgressBar(
            leave_epoch_progress=False, update_per_second=1, show_epoch_progress=False
        )

        history = self.model.fit(
            self.window.train,
            epochs=MAX_EPOCHS,
            validation_data=self.window.valid,
            callbacks=[callback_]
            if self.verbose == "show"
            else [callback_, progress_bar],
            verbose=1 if self.verbose == "show" else 0,
        )

        return history

    def evaluate(self) -> Tuple[Dict[str, List[float]]]:
        """EValuate the trained model on valid and test datasets
        from the window attribute."""
        valid_ = self.model.evaluate(self.window.valid)

        test_ = self.model.evaluate(self.window.test, verbose=0)

        return valid_, test_

    def plot(self, **kwargs) -> None:
        """Plot the model' prediction on example windows."""
        self.window.plot(self.model, self.name, save=self.name, **kwargs)
        
    def get_config(self):
        """Used to be able to laod the model later."""
        base_config = self.model.get_config()
        return base_config#{**base_config, "num_classes": self.num_classes}