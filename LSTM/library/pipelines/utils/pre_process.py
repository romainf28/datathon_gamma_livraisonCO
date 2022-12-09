from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ...data.pipeline import do_datasets

rename_map = {
    'DATE': 'date',
    'SCC_temp_degreC': 'SCC_temp',
    'boiler_outlet_temp_degreC': 'boiler_temp',
    'boiler_transfer_W_K_m2': 'boiler_transfer',
    'average_SCC_temp_degreC_last_cleaning_campaign': 'SCC_temp_cleaning',
    'time_from_last_campaign': 'uptime',
    'last_cleaning_campaign_duration': 'cleaning_time',
    'total_odslime_flow_kg_since_last_cleaning': 'acc_odslime',
    'total_T105_flow_kg_since_last_cleaning': 'acc_T105',
    'total_solid_waste_flow_kg_since_last_cleaning': 'acc_solidwaste',
    'total_special_flow_kg_since_last_cleaning': 'acc_specialflow',
    'total_ods_flow_kg_since_last_cleaning': 'acc_ods',
    'total_sludge_flow_kg_since_last_cleaning': 'acc_sludge',
    'total_Aqueous_flow_kg_since_last_cleaning': 'acc_aqueous',
    'total_organic_acid_flow_kg_since_last_cleaning': 'acc_organicacid',
    'total_ew_flow_kg_since_last_cleaning': 'acc_ew',
    'total_thermal_power_kwh_since_last_cleaning': 'acc_thpower',
    'has_cleaning_campaign': 'filter_',
}


def pre_process(
        df: pd.DataFrame,
        outliers: str = 'replacing',
        date_range: Tuple[str] = None,
        format_: str = '%Y-%m-%d %H:%M:%S',
        df_mean: pd.Series = None,
        df_std: pd.Series = None,
        MGA_k: int = 5
) -> Tuple[pd.DataFrame]:
    """Do the datasets.

    df_mean and df_std: optional parameters to normalize with given sets."""
    cols = [c for c in df.columns if '+' not in c and '-' not in c] #and 'std' not in c]
    df = df[cols]

    df = df.rename(columns=rename_map)
    df['date'] = pd.to_datetime(df['date'], format=format_).astype('datetime64[s]')

    # date range
    if date_range:
        mask = (df['date'] < date_range[1]) & (df['date'] >= date_range[0])
        df = df.loc[mask]

    #df = df.drop(df.filter(like='acc', axis=1).columns, axis=1)

    # fill missing val
    df = df.set_index('date').reindex(pd.date_range(df.date.min(), df.date.max(), freq='H')).rename_axis(
        'date').reset_index()
   
    is_NaN = df.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    df.loc[row_has_NaN, 'filter_'] = 0
   

    #remove_cols = ['cut_off_time_hour0', 'cut_off_time_hour', 'SCC_temp', 'cut_off_time_hour', 'thermal_power_kwh',
    #               'special_flow_kg_hour', 'cleaning_time', 'ods_flow_kg_hour', 'T105_flow_kg_hour']
    #df = df.drop(remove_cols, axis=1)

    outliers_ = np.abs(
        stats.zscore(df[df.filter_ == 0].drop(['filter_', 'date'], axis=1)) < 3).all(axis=1)
    
    ##Former filtering metho
    #if outliers == 'replacing':
     #   for col in df.columns:
     #       if np.issubdtype(df[col].dtype, np.number) and col != 'has_cleaning_campaign':
                # set nan to not take into account bad values
     #           df.loc[outliers_[outliers_ == False].index, col] = np.NaN
    
    print("{} outliers detected".format(outliers_[outliers_ == False].count()))


    index_ = df.filter_ == 1
    index_ = index_[index_ == True].index

    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number) and col != 'filter_':
            # set nan to not take into account bad values
            df.loc[index_, col] = np.NaN
            # gaussian filtering
            df[col] = df[col].rolling(window=MGA_k, min_periods=1, center=False, win_type='gaussian').mean(std=3)
            

    # add interpolation method to fill missing values in the replacing method
    df = df.interpolate(method='pad')

    return do_datasets(
        data=df,
        unscale_cols=['date', 'filter_'],
        df_mean=df_mean,
        df_std=df_std
    )