"""Deprecated. This file contains basic processing and data loaders making up first steps of the pipeline."""
from typing import List, Tuple

import pandas as pd


def data_splitter(data: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """Split the given dataset into train, test and valid dataset."""
    n = len(data)
    df_train = data[0:int(n * 0.7)]
    df_valid = data[int(n * 0.7):int(n * 0.9)]
    df_test = data[int(n * 0.9):]

    return df_train, df_valid, df_test