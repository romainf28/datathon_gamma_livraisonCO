"""This files contains window_from_data and training_model pipelines."""
from .utils.pre_process import pre_process
from ..data.window.generator import WindowGenerator
from ..models.base.template import Predictor
from typing import Any, List

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

import logging


logger = logging.getLogger(__name__)

def window_from_data(
    file_path: str = "Data/processed_data_New.csv",
    time_format: str = "%Y-%m-%d %H:%M:%S",
    batch_size: int = 4,
    SAMPL_RT: int = 6,
    INSTEPS_DAYS: int = 7,
    OUTSTEPS_DAYS: int = 21,
    label_columns: List[str] = ["boiler_temp", "boiler_transfer"],
    features_col: List[str] = [
        "boiler_temp",
        "boiler_transfer",
        "odslime_flow_kg_hour",
        "solid_waste_flow_kg_hour",
        "ew_flow_kg_hour",
        "organic_acid_flow_kg_hour",
        "SCC_temp_cleaning",
    ],
    reload_data: bool=False
):
    """Prepare the data for ml, do the logs and register parameters. 
    
    reload_data parameters: wether should store new dump of data std and mean
    use for scaling data back in forecast and windows. You may set it to true 
    if you use new features.
    
    Return a window object define in the package."""
    
    if reload_data:
        df_mean = None
        df_std=None
    else:
        with open('saved_data/data_std' , 'rb') as f:
            data = pickle.load(f)
            logger.info('Retrevied mean and std')
            df_mean = data['df_mean']
            df_std = data['df_std']
    
    df, df_mean, df_std = pre_process(pd.read_csv(file_path), format_=time_format, df_mean=df_mean, df_std=df_std)
    

    # save the data parameters for the prediction pipeline
    if reload_data:
        with open("saved_data/data_std", "wb") as files:
            pickle.dump({"df_mean": df_mean, "df_std": df_std}, files)
            logger.info('Dumped mean and std')

    # number of windows
    def count_windows(interval, df):
        stop_indexes = df[df.filter_ == 1].index
        id_s = df.index.difference(df[df.filter_ == 1].index)

        counter = 0
        for slicer in range(len(id_s) - interval):
            if id_s[slicer + interval] - id_s[slicer] == interval:
                counter += 1
        return counter

    interval = 24 * (7 + 21)

    logger.warning(
        "{} windows of {} hours can be extracted".format(
            count_windows(interval, df), interval
        )
    )

    
    INSTEPS = int(INSTEPS_DAYS * 24 / SAMPL_RT)
    OUTSTEPS = int(OUTSTEPS_DAYS * 24 / SAMPL_RT)
    
    return WindowGenerator(
        input_width=INSTEPS,
        label_width=OUTSTEPS,
        shift=OUTSTEPS,
        df=df.drop("date", axis=1),
        df_mean=df_mean,
        df_std=df_std,
        feature_columns=features_col,
        filter_col="filter_",
        label_columns=label_columns,
        batch_size=batch_size,
        sampling_rate=SAMPL_RT,
    )

def training_model(
    model_generator: Any = None,
    wrapper: Any=None,
    optimizer:  tf.optimizers = tf.optimizers.Adam(),
    window_: Any=None,
    loss_mod: str = 'classic',
    model_name: str = 'pipeline_cli',
    file_path: str = 'Data/processed_data_New.csv',
    time_format: str = '%Y-%m-%d %H:%M:%S',
    batch_size: int = 4,
    SAMPL_RT: int = 6,
    INSTEPS_DAYS: int = 7,
    OUTSTEPS_DAYS: int = 21,
    num_labels: int = 2,
    MAX_EPOCHS: int= 50,
    plot: bool=True,
    verbose: str= 'hide',
    *args,
    **kwargs,
):
    """Kwargs contain optimizer instance, MAX_EPOCHS and callback.
    
    
    Return val_performacne, test_perfomace and the Predictor object.
    Keras functionnal model is the model attribute of this object: Predictor.model"""
    
    if not model_generator:
        OUTSTEPS = int(OUTSTEPS_DAYS * 24 / SAMPL_RT)
        model_generator = lambda: tf.keras.Sequential([
            tf.keras.layers.LSTM(int(OUTSTEPS*num_labels*2.5), return_sequences=False),
            tf.keras.layers.Dense(OUTSTEPS*num_labels),
            tf.keras.layers.Reshape([OUTSTEPS, num_labels])
        ])
        
    if not window_:
        window_ = window_from_data(
            file_path=file_path,
            time_format=time_format,
            batch_size=batch_size,
            SAMPL_RT=SAMPL_RT,
            INSTEPS_DAYS=INSTEPS_DAYS,
            OUTSTEPS_DAYS=OUTSTEPS_DAYS
        )
        
    cli_model = Predictor(
    model_name, 
    window=window_, 
    model_generator=model_generator,
    MAX_EPOCHS=MAX_EPOCHS,
    loss_mod=loss_mod,
    optimizer=optimizer,
    wrapper=wrapper,
    verbose=verbose,
    *args,
    **kwargs,
    )

    val_performance, performance = cli_model.evaluate()
    
    #save training curve
    if plot:
        cli_model.plot()
    
    return val_performance, performance, cli_model