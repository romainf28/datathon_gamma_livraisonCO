"""File used for research notebook. Not required for production."""
from .pipeline_utils.functions import data_splitter
from .pipeline_utils.engineering import time_engineering
from typing import Tuple, List, Union

import pandas as pd


def do_datasets(
        data: pd.DataFrame,
        unscale_cols: List[str] = [],
        df_mean: pd.Series = None,
        df_std: pd.Series = None,
        **kwargs,
) -> Tuple[Union[pd.DataFrame, pd.Series]]:
    """Data pipeline. Return the three datasets."""

    #df = time_engineering(df_indexed=data, time_set=True) not used in this
    #project
    df = data
    transform_cols = df.columns.difference(unscale_cols)

    # we are proceeding real application
    if not (isinstance(df_mean, pd.Series) and isinstance(df_std, pd.Series)):
        df_mean = df.loc[:, transform_cols].mean()
        df_std = df.loc[:, transform_cols].std()

    # normalize the data according to df_train
    df.loc[:, transform_cols] = (df.loc[:, transform_cols] - df_mean) / df_std

    return df, df_mean, df_std
