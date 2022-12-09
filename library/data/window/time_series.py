"""Custom low-level tensorflow API functions to create time series dataset from dataframe 'array'.
We set in this folder the float64 parameter. You may consider switching it to float32 to lower memory
usage."""

import tensorflow.compat.v2 as tf

import numpy as np
from tensorflow.python.util.tf_export import keras_export

import pandas as pd


def timeseries_dataset_from_array(
    data_: pd.DataFrame,
    targets,
    sequence_length,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=128,
    filter_: str = "filter_",
    shuffle=False,
    seed=None,
    start_index=None,
    end_index=None,
    index_diff=-1,
):
    """Based on officiel tf keras function. Supercharged to not batch and compute much faster.

    Starting index will be computed from the filter_ property of the dataframe. No window should
    contrain a filter_ == 1 value.

    Convert fitered df to a batch dataset (tensor) ready to be used by the model.

    index_diff param is non equal to -1 when we are dealing with cumulated variables.
    for instance, the total_XX_since last cleaning window [12,12,13,14] will be [12,0,1,1].

    Return a batch dataset with the last date."""
    data = np.array(data_, dtype=np.float64)
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data)

    # Determine the lowest dtype to store start positions (to lower memory usage).
    num_seqs = end_index - start_index - (sequence_length * sampling_rate) + 1
    if targets is not None:
        num_seqs = min(num_seqs, len(targets))
    if num_seqs < 2147483647:
        index_dtype = "int32"
    else:
        index_dtype = "int64"
    # arnaud: custom starting position
    interval = sampling_rate * sequence_length
    stop_indexes = data_[data_[filter_] == 1].index
    id_s = data_.index.difference(data_[data_[filter_] == 1].index)

    start_positions = []
    for slicer in range(len(id_s) - interval):
        # check if continuous
        if id_s[slicer + interval] - id_s[slicer] == interval:
            start_positions.append(id_s[slicer])
    start_positions = np.array(start_positions, dtype=index_dtype)
    # original code:
    # np.arange(0, num_seqs, sequence_stride, dtype=index_dtype)
    if shuffle:
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(start_positions)

    sequence_length = tf.cast(sequence_length, dtype=index_dtype)
    sampling_rate = tf.cast(sampling_rate, dtype=index_dtype)

    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()

    # For each initial window position, generates indices of the window elements
    indices = tf.data.Dataset.zip(
        (tf.data.Dataset.range(len(start_positions)), positions_ds)
    ).map(
        lambda i, positions: tf.range(  # pylint: disable=g-long-lambda
            positions[i], positions[i] + sequence_length * sampling_rate, sampling_rate
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = sequences_from_indices(data, indices, start_index, end_index, index_diff)
    if targets is not None:
        indices = tf.data.Dataset.zip(
            (tf.data.Dataset.range(len(start_positions)), positions_ds)
        ).map(lambda i, positions: positions[i], num_parallel_calls=tf.data.AUTOTUNE)
        target_ds = sequences_from_indices(targets, indices, start_index, end_index)
        dataset = tf.data.Dataset.zip((dataset, target_ds))
    if shuffle:
        # Shuffle locally at each iteration
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # .batch(batch_size)
    return dataset


def sequences_from_indices(array, indices_ds, start_index, end_index, index_diff=-1):
    """Utility function used to generate the batch datasets.
    The cumulated variables transformation (cf doc of df_to_tensor) is computed in the
    else section.
    Operation on tensor slices is not allowed with tensorflow, thus we have to pass through
    numpy."""
    if index_diff == -1:
        dataset = tf.data.Dataset.from_tensors(array[start_index:end_index])
    else:
        array_ = array[start_index:end_index]

        diff_ = np.concatenate(
            (
                array_[0, index_diff:][np.newaxis, :],
                np.diff(array_[:, index_diff:], axis=0),
            )
        )
        dataset = tf.data.Dataset.from_tensors(
            np.concatenate((array_[:, :index_diff], diff_), axis=1)
        )
        del diff_

    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
        lambda steps, inds: tf.gather(
            steps, inds
        ),  # pylint: disable=unnecessary-lambda
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset
