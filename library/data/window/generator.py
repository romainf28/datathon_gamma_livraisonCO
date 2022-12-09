"""This file contrain the windowgenerator class, making up the base of the time series problem."""

from math import sqrt
from typing import Dict, Any, Tuple, List
from random import random
from .time_series import timeseries_dataset_from_array

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class WindowGenerator:
    """The WindowGenerator class will contain the windowed used to train the model.

    Use the train, test and valid properties to train and review models.

    By default, the windows will be shuffled through the subsets. Set shuffle_windows to
    False if you don't want that.

    The window will be a tuple going from feature_columns to label_columns. It means you
    don't need to remove from the given dataframe unuseful columns.

    The batch size is set here.
    Example print with default parameters:

    Total window size: 112
    Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27]
    Label indices: [ 28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45
      46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
      64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81
      82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99
     100 101 102 103 104 105 106 107 108 109 110 111]
    Label column name(s): ['boiler_temp', 'boiler_transfer']
    """

    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        df: pd.DataFrame,
        df_mean: pd.Series,
        df_std: pd.Series,
        feature_columns: List[str] = None,
        label_columns: List[str] = None,
        batch_size: int = 1,
        sampling_rate: int = 1,
        filter_col: str = "filter_",
        example_reload: bool = True,
        shuffle_windows: bool = True,
        diff_acc: bool = True,
        train_split: float = 0.7
    ) -> None:
        self.batch_size = batch_size
        self.shuffle_windows = shuffle_windows
        
        self.train_split = train_split
        # Store the raw data.
        self.df = df

        self.df_mean = df_mean
        self.df_std = df_std

        self.sampling_rate = sampling_rate

        # Work out the label column indices.
        # diff_acc: proceed to the difference everywhere or not
        if diff_acc:
            diff_features = [
                col
                for col in feature_columns
                if "std" in col or "total" in col or "acc" in col or "uptime" in col
            ]
            raw_features = [col for col in feature_columns if col not in diff_features]
            self.feature_columns = raw_features + diff_features
            self.index_diff = len(raw_features)

        else:
            self.index_diff = -1
            self.feature_columns = feature_columns

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(self.label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(df.columns)}
        # use for plotting
        if label_columns and self.label_columns[0] in feature_columns:
            self.label_input_indice = self.feature_columns.index(label_columns[0])
        else:
            self.label_input_indice = -1

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        self.total_indices = np.arange(self.total_window_size)

        self.input_slice = slice(0, input_width)
        self.input_indices = self.total_indices[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = self.total_indices[self.labels_slice]

        self.filter_col = filter_col

        # reload the example window or not
        self.example_reload = example_reload

    def __repr__(self):
        """Example output:
        Total window size: 112
        Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
         24 25 26 27]
        Label indices: [ 28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45
          46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
          64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81
          82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99
         100 101 102 103 104 105 106 107 108 109 110 111]
        Label column name(s): ['boiler_temp', 'boiler_transfer']
    
        """
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features) -> Tuple[tf.Tensor]:
        """Split the computed window into features and labels.
        The corresponding timestamps are given by rep.
        
        Return Tuple of tensors."""

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.feature_columns is not None:
            inputs = tf.stack(
                [
                    inputs[:, :, self.column_indices[name]]
                    for name in self.feature_columns
                ],
                axis=-1,
            )

        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data) -> Any:
        """Make the dataset with the custom timeseries_dataset_from_array.
        """
        # data = np.array(data, dtype=np.float32)
        ds = timeseries_dataset_from_array(
            data_=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            sampling_rate=self.sampling_rate,
            filter_=self.filter_col,
            seed=42,
            # used for shuffle only
            shuffle=self.shuffle_windows,
            batch_size=self.batch_size,
            index_diff=self.index_diff,
        )
        # old method. Implemented into timeseries dataset much fasteriwth precomuputed starting index

        # def filter_(x):
        #    return tf.reduce_all(tf.math.equal(x[:, self.filter_indice], 0))

        # mask_template = [True for _ in range(len(self.column_indices)+1)]
        # mask_template[16] = False
        # ds = ds.filter(filter_)
        # apply the masl to remove the feature has_ ...  and rebatch
        # ds = ds.map(lambda x: tf.boolean_mask(x, np.array(mask_template), axis=1)).batch(
        #    self.batch_size
        # )

        ds = ds.batch(self.batch_size)
        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        """This property will get the dataset if already computed. If not, it will compute its length.
        
        Default: 70% of initial dataset, shuffle randomly."""
        dataset = getattr(self, "_dataset", None)
        if dataset is None:
            dataset = self.make_dataset(data=self.df)
            # And cache it for next time
            self._dataset = dataset
            # only executed once, don t need the tf function decorator
            dataset_len = dataset.reduce(tf.constant(0), lambda x, _: x + 1).numpy()
            self._dataset_len = dataset_len
        else:
            # len has already been computed by reduce function
            dataset_len = getattr(self, "_dataset_len", None)
            
        return dataset.take(int(self.train_split * dataset_len))

    @property
    def valid(self):
        """This property will get the dataset if already computed. If not, it will compute its length.
        
        Default: 20% of initial dataset, shuffle randomly."""
        dataset = getattr(self, "_dataset", None)
        if dataset is None:
            dataset = self.make_dataset(data=self.df)
            # And cache it for next time
            self._dataset = dataset
            dataset_len = dataset.reduce(tf.constant(0), lambda x, _: x + 1).numpy()
            self._dataset_len = dataset_len
        else:
            dataset_len = getattr(self, "_dataset_len", None)

        if self.train_split == 1:
            return dataset.take(int(self.train_split * dataset_len))
        
        return dataset.skip(
            int(0.7 * dataset_len)
        ).take(int(0.2 * dataset_len))

    @property
    def test(self):
        """This property will get the dataset if already computed. If not, it will compute its length.
       
        Default: 10% of initial dataset, shuffle randomly."""
        dataset = getattr(self, "_dataset", None)
        if dataset is None:
            dataset = self.make_dataset(data=self.df)
            # And cache it for next time
            self._dataset = dataset
            dataset_len = dataset.reduce(tf.constant(0), lambda x, _: x + 1).numpy()
            self._dataset_len = dataset_len
        else:
            dataset_len = getattr(self, "_dataset_len", None)

        if self.train_split == 1:
            return dataset.take(int(self.train_split * dataset_len))
        
        return dataset.skip(int(0.9 * dataset_len))

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        if self.example_reload:
            return next(iter(self.train))
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
    
    def plot(
        self,
        model: Any = None,
        model_name: str = "",
        save: str = None,
    ) -> None:
        """Note to Tony from Arnaud: This function is broken if you have only one output. It is not difficult to solve but
        I have to rush the project. Super useful if you have to do a plot for a report.
        Take care.
        """
        # Dict
        inputs, labels = self.example
        num_col = min(4, self.batch_size)

        plt.figure(figsize=(12, 8))

        fig, axs = plt.subplots(len(self.label_columns), num_col, figsize=(30, 10))
        fig.suptitle(model_name)

        for i_, label in enumerate(self.label_columns):
            plot_col_index = self.column_indices[label]

            if label in self.feature_columns:
                label_input_indice = self.feature_columns.index(label)
            else:
                label_input_indice = -1
            for n in range(num_col):
                working_ax = axs[i_, n]
                working_ax.set(ylabel=f"{label}")
                if self.label_input_indice != -1:
                    working_ax.plot(
                        self.input_indices,
                        inputs[n, :, label_input_indice] * self.df_std[label]
                        + self.df_mean[label],
                        label="Inputs",
                        marker=".",
                        zorder=-10,
                    )

                if self.label_columns:
                    label_col_index = self.label_columns_indices.get(label, None)
                else:
                    label_col_index = plot_col_index
                if label_col_index is None:
                    continue

                working_ax.scatter(
                    self.label_indices,
                    labels[n, :, label_col_index] * self.df_std[label]
                    + self.df_mean[label],
                    edgecolors="k",
                    label="Labels",
                    c="#2ca02c",
                    s=64,
                )
                if model is not None:
                    predictions = model(inputs)
                    working_ax.scatter(
                        self.label_indices,
                        predictions[n, :, label_col_index] * self.df_std[label]
                        + self.df_mean[label],
                        marker="X",
                        edgecolors="k",
                        label="Predictions",
                        c="#ff7f0e",
                        s=64,
                    )

                if n == 0:
                    working_ax.legend()
        if save:
            fig.savefig("outputs/{}.png".format(save))

        # plt.xlabel("Time [h]")
