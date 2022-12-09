import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import (
    read_csv_files,
    filter_nodes,
    convert_dates_to_datetime,
    add_holidays,
    add_weather_data,
    set_indexes_for_timeseries,
    get_train_test,
    fill_missing_values,
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from library import WindowGenerator, do_datasets, training_model
import tensorflow as tf


def mechant_romain():
    DATA_DIR = "./data"
    filenames = ["champs-elysees.csv", "convention.csv", "saints-peres.csv"]
    dfs = read_csv_files(DATA_DIR, filenames)
    TIME_WINDOW = 14
    df = pd.concat(dfs)
    df_filtered = filter_nodes(df)
    df_filtered = df_filtered.drop(
        ["Libelle noeud amont", "Libelle noeud aval"], axis=1
    )
    df_filtered["Jour de la semaine"] = pd.to_datetime(
        df_filtered["Date et heure de comptage"]
    ).dt.dayofweek
    df_filtered = pd.concat(
        [
            df_filtered,
            pd.get_dummies(
                df_filtered["Jour de la semaine"], prefix="Jour de la semaine"
            ),
        ],
        axis=1,
    ).drop(columns=["Jour de la semaine"])

    df_filtered.sample(10)
    df_filtered["Date"] = pd.to_datetime(
        df_filtered["Date et heure de comptage"]
    ).dt.date
    add_holidays(df_filtered, DATA_DIR)
    df_filtered = add_weather_data(df_filtered, DATA_DIR)

    traffic_state_encoding = {
        "Inconnu": 0,
        "Fluide": 1,
        "Pré-saturé": 2,
        "Saturé": 3,
        "Bloqué": 4,
    }
    df_filtered["Etat trafic"] = df_filtered["Etat trafic"].map(traffic_state_encoding)
    df_train, df_test = get_train_test(df_filtered)

    df_train_ce = df_train[df_train["filename"] == "champs-elysees.csv"]
    df_train_sts = df_train[df_train["filename"] == "saints-peres.csv"]
    df_train_conv = df_train[df_train["filename"] == "convention.csv"]

    df_test_ce = df_test[df_test["filename"] == "champs-elysees.csv"]
    df_test_sts = df_test[df_test["filename"] == "saints-peres.csv"]
    df_test_conv = df_test[df_test["filename"] == "convention.csv"]

    dfs_train = [df_train_ce, df_train_sts, df_train_conv]
    dfs_test = [df_test_ce, df_test_sts, df_test_conv]

    return dfs_train, dfs_test


def mechant_arnaud(df_: pd.DataFrame):
    # df_ = dfs_train[0]

    date_time = df_["Date et heure de comptage"]
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    week = 7 * day

    import numpy as np

    df_["daySin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df_["weekSin"] = np.sin(timestamp_s * (2 * np.pi / week))

    df_ = df_[
        [
            "Débit horaire",
            "Taux d'occupation",
            "Etat trafic",
            "daySin",
            "weekSin",
            "humidity",
            "Durée avant les prochaines vacances scolaires",
        ]
    ]
    df_["filter"] = df_.isna().any(axis=1)
    return df_


def mechant_ml(df_: pd.DataFrame):
    std_1 = df_["Débit horaire"].std()
    std_2 = df_["Taux d'occupation"].std()

    df_["Durée avant les prochaines vacances scolaires"] = df_[
        "Durée avant les prochaines vacances scolaires"
    ].dt.days

    df, df_mean, df_std = do_datasets(data=df_, unscale_cols=["filter"])

    WIN_LENGTH = 24 * 5
    batch_size = 128

    window = WindowGenerator(
        input_width=int(2.5 * WIN_LENGTH),
        label_width=WIN_LENGTH,
        shift=24 * 5,
        df=df,
        df_mean=df_mean,
        df_std=df_std,
        feature_columns=[
            "Débit horaire",
            "Taux d'occupation",
            "Etat trafic",
            "daySin",
            "weekSin",
            "humidity",
            "Durée avant les prochaines vacances scolaires",
        ],
        filter_col="filter",
        label_columns=["Débit horaire", "Taux d'occupation"],
        batch_size=batch_size,
        train_split=0.7,
        sampling_rate=1,
        shuffle_windows=True,
    )

    OUTSTEPS = WIN_LENGTH
    num_labels = 2

    lstm_generator = lambda multipl: lambda: tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                int(OUTSTEPS * num_labels * multipl), return_sequences=False
            )
            for _ in range(1)
        ]
        + [
            tf.keras.layers.Dense(OUTSTEPS * num_labels),
            tf.keras.layers.Reshape([OUTSTEPS, num_labels]),
        ]
    )

    optimizer = tf.optimizers.Adam()
    callback_mod = "EarlyStopping"
    loss_mod = "mae"
    model_generator = lstm_generator(2)

    model_name = "final_model"

    val_performance = {}
    cli = {}

    val_performance["lstm_4"], _, cli["lstm_4"] = training_model(
        model_generator=lstm_generator(4),
        window_=window,
        model_name="lstm",
        MAX_EPOCHS=200,
        loss_mod=loss_mod,
        optimizer=optimizer,
        plot=False,
        verbose="show",
        no_clone=True,
    )

    prediction = cli["lstm"].model.predict(window.valid)
    target = tf.concat([windo[1] for windo in window.valid], 0)

    prediction_1 = prediction[:, :, 0]
    target_1 = target[:, :, 0]
    prediction_2 = prediction[:, :, 1]
    target_2 = target[:, :, 1]

    m_1 = tf.keras.metrics.RootMeanSquaredError()
    m_1.update_state(prediction_1, target_1)

    m_2 = tf.keras.metrics.RootMeanSquaredError()
    m_2.update_state(prediction_2, target_2)

    kpi_1 = m_1.result().numpy() * std_1
    kpi_2 = m_2.result().numpy() * std_2

    return cli["lstm"].model, (kpi_1, kpi_2)


def main():
    df_train, df_test = mechant_romain()
    rue = ["ce", "sts", "conv"]
    for i, rue in zip(range(len(df_train)), rue):
        df_ = df_train[i]
        df_ = mechant_arnaud(df_)
        model, kpis = mechant_ml(df_)
        print(f"{rue} : {kpis[0]}_{kpis[1]}")
        model.save(f"{rue}_{kpis[0]}_{kpis[1]}")
        
        
if __name__ == "__main__":
    main()
