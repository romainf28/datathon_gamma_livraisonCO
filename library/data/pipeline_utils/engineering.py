"""Deprecated : not used in the ecospace project. Could be useful for any other project with seasonality in the data.
This file contains functions aiming at creating new features for the ML models."""
import pandas as pd
import numpy as np


def time_engineering(df_indexed: pd.DataFrame, time_set: bool = False) -> pd.DataFrame:
    """Create a sninus transformation of time: daily and yearly periodicity."""
    # fix values
    day = 24 * 60 * 60
    year = 365.2425 * day

    # convert all dates to seconds
    timestamp_s = df_indexed["date"].map(pd.Timestamp.timestamp)
    # compute their sin transform
    day_sin = np.sin(timestamp_s * (2 * np.pi / day))
    year_sin = np.sin(timestamp_s * (2 * np.pi / year))
    # add new features to dataframe
    data = pd.concat([df_indexed, day_sin, year_sin], axis=1)
    # rename added columns
    data.columns = list(data.columns)[:-2] + ["days_sin", "years_sin"]
    # we don t need the date col anymore
    # data.drop(columns=["date"], inplace=True)

    return data
