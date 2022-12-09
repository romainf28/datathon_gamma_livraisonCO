"""This files contains forecast pipeline used in the deployed app on google cloud server."""
import pickle
from typing import List, Tuple

import pandas as pd
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf

from .utils.pre_process import pre_process
from .utils.to_tensors import df_to_tensor

logger = logging.getLogger(__name__)

def forecast(
        df: pd.DataFrame,
        model_path: str = 'experimental_models/final',
        SAMPL_RT: int = 6,
        input_len: int = 28,
        target_len: int = 84,
        date_range: Tuple = None,
        time_format: str = '%Y-%m-%d %H:%M:%S',
        label_cols: List[str] = ['boiler_temp'],
        feature_columns=['boiler_temp', 
                         'acc_thpower', 
                         'std_thermal_power_kwh_since_last_cleaning', 
                         'std_solid_waste_flow_kg_since_last_cleaning', 
                         'std_T105_flow_kg_since_last_cleaning', 
                         'SCC_temp', 
                         'std_odslime_flow_kg_since_last_cleaning', 
                         'std_ew_flow_kg_since_last_cleaning', 
                         'std_organic_acid_flow_kg_since_last_cleaning', 
                         'boiler_transfer', 
                         'acc_T105', 
                         'ew_flow_kg_hour', 
                         'std_ods_flow_kg_since_last_cleaning', 
                         'Aqueous_flow_kg_hour', 
                         'uptime', 
                         'SCC_temp_cleaning', 
                         'acc_solidwaste', 
                         'cleaning_time'],
        diff_acc: bool=True,
) -> Tuple[pd.DataFrame]:
    """
    Forecast function used for production (requires a trained model).
    
    Load a dataframe in the function call, it will automatically do the forecast
    starting with the last possible date. This date is computed with the input len.
    
    
    return
    """
    # use pickle to load df_std and mean
    with open('saved_data/data_std', 'rb') as file:
        data_pretrain = pickle.load(file)

    df_train_mean, df_train_std = data_pretrain['df_mean'], data_pretrain['df_std']
    df_p, _, _ = pre_process(
        df=df, 
        df_mean=df_train_mean, 
        df_std=df_train_std, 
        format_=time_format, 
        date_range=date_range
    )
    if diff_acc:
            diff_features = [col for col in feature_columns if 'std' in col or 'total' in col or 'acc' in col or 'uptime' in col]
            raw_features= [col for col in feature_columns if col not in diff_features]
            feature_columns = raw_features+diff_features
            index_diff = len(raw_features)
            
    else:
        index_diff=-1
        feature_columns = feature_columns
                
    tensor, date_init = df_to_tensor(df=df_p,
                                     features_col=feature_columns,
                                     target_len=input_len,
                                     SAMPL_RT=SAMPL_RT,
                                     index_diff=index_diff
                                     )

    model = tf.keras.models.load_model(model_path, compile=False)
    logger.info(f"Model Loaded from {file}")
    prediction = model(next(iter(tensor)))

    results_dict = {id_: tf.reshape(prediction[:, :, k], -1) * df_train_std[id_] + df_train_mean[id_] for k, id_ in
                    enumerate(label_cols)}
    results_dict['date'] = [date_init + pd.Timedelta(hours=SAMPL_RT) * (k + 1) for k in range(target_len)]
    return pd.DataFrame.from_dict(results_dict).set_index('date'), (df_p, date_init + pd.Timedelta(hours=SAMPL_RT))
