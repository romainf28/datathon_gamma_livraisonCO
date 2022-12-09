from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf

logger = logging.getLogger(__name__)


def sequences_from_indices(array, indices_ds, start_index, end_index, index_diff =-1):
    """Utility function used to generate the batch datasets.
    The cumulated variables transformation (cf doc of df_to_tensor) is computed in the
    else section.
    Operation on tensor slices is not allowed with tensorflow, thus we have to pass through
    numpy.
    """
    # we can work with int32 index    
    if index_diff == -1:
        dataset = tf.data.Dataset.from_tensors(array[start_index:end_index])
    else:    
        array_ = array[start_index:end_index]
        
        diff_ = np.concatenate((array_[0, index_diff:][np.newaxis, :], np.diff(array_[:,index_diff:], axis=0)))
        dataset = tf.data.Dataset.from_tensors(np.concatenate((array_[:,:index_diff], diff_), axis=1))
        del diff_
    
    
    
    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
        lambda steps, inds: tf.gather(
            steps, inds
        ),  # pylint: disable=unnecessary-lambda
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset


def df_to_tensor(
        df: pd.DataFrame,
        features_col: List[str],
        target_len: int = 28,
        SAMPL_RT: int = 6,
        index_diff=-1
) -> Tuple[Any, Any]:
    """Convert fitered df to a batch dataset (tensor) ready to be used by the model.

    index_diff param is non equal to -1 when we are dealing with cumulated variables.
    for instance, the total_XX_since last cleaning window [12,12,13,14] will be [12,0,1,1].
    
    Return a batch dataset with the last date. """
    end_index = len(df)
    start_index = end_index - target_len * SAMPL_RT
    #shift to start from midnight, only work with sampling rate of one hour !!!!!!
    hour_ = df.loc[start_index, 'date'].hour
    start_index -= hour_
    end_index -= hour_

    start_after_index = df[df.filter_ == 1].index[-1]
    if start_index <= start_after_index:
        logger.warning(f"Invalid data in the extracted window")
        #raise NameError('Invalid data: can t extract the window properly')

    data_review = np.array(df[features_col], dtype=np.float64)

    end_index = tf.cast(end_index, dtype="int32")
    sequence_length = end_index - start_index + 1

    interval = SAMPL_RT * sequence_length

    start_positions = np.array([tf.cast(0, dtype="int32")], dtype="int32")

    sequence_length = tf.cast(sequence_length, dtype="int32")
    sampling_rate = tf.cast(SAMPL_RT, dtype="int32")

    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()

    indices = tf.data.Dataset.from_tensors(
        tf.range(
            start_index,
            end_index, sampling_rate
        )
    )
    if index_diff == -1:
        dataset_review = tf.data.Dataset.from_tensors(data_review[tf.cast(0, dtype="int32"):end_index])
        
    else:
        array_ = data_review[tf.cast(0, dtype="int32"):end_index]
        
        diff_ = np.concatenate((array_[0, index_diff:][np.newaxis, :], np.diff(array_[:,index_diff:], axis=0)))
        dataset_review = tf.data.Dataset.from_tensors(np.concatenate((array_[:,:index_diff], diff_), axis=1))
        del diff_
        
        
    dataset_review = tf.data.Dataset.zip((dataset_review.repeat(), indices)).map(
        lambda steps, inds: tf.gather(
            steps, inds
        )
    )

    dataset_review = dataset_review.prefetch(tf.data.AUTOTUNE).batch(1)
    logger.info(f"Ready to forecast")
    return dataset_review, df.loc[next(iter(indices))[-1].numpy(), 'date']
