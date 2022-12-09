"""Functions used to perform slice on multiIndex DataFrames. This file is deprecated. Functions where used in the Load control project 
with multi index (several models to train on  dispatch data at the same time)."""
import pandas as pd


def slicer(df: pd.DataFrame, slice: slice) -> pd.DataFrame:
    """Perform a slice on multiindex dataframe. Same slice on second level."""
    # we have to do this trick

    return df.groupby(level=0).apply(
        lambda group: group.iloc[slice]
    ).reset_index(level=0, drop=True)


def slice_array(df: pd.DataFrame, slice: slice) -> pd.DataFrame:
    """Slice and return as array [site, sample, feature]."""
    df_sliced = slicer(df, slice)
    df_np = df_sliced.groupby(level=0).apply(lambda x: x.values).values

    return np.stack(df_np)