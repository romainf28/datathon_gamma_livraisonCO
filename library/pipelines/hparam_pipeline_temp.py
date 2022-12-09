from .training_pipeline import training_model, window_from_data
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

import logging
import os


logger = logging.getLogger(__name__)


rename_map = {
    "DATE": "date",
    "SCC_temp_degreC": "SCC_temp",
    "boiler_outlet_temp_degreC": "boiler_temp",
    "boiler_transfer_W_K_m2": "boiler_transfer",
    "average_SCC_temp_degreC_last_cleaning_campaign": "SCC_temp_cleaning",
    "time_from_last_campaign": "uptime",
    "last_cleaning_campaign_duration": "cleaning_time",
    "total_odslime_flow_kg_since_last_cleaning": "acc_odslime",
    "total_T105_flow_kg_since_last_cleaning": "acc_T105",
    "total_solid_waste_flow_kg_since_last_cleaning": "acc_solidwaste",
    "total_special_flow_kg_since_last_cleaning": "acc_specialflow",
    "total_ods_flow_kg_since_last_cleaning": "acc_ods",
    "total_sludge_flow_kg_since_last_cleaning": "acc_sludge",
    "total_Aqueous_flow_kg_since_last_cleaning": "acc_aqueous",
    "total_organic_acid_flow_kg_since_last_cleaning": "acc_organicacid",
    "total_ew_flow_kg_since_last_cleaning": "acc_ew",
    "total_thermal_power_kwh_since_last_cleaning": "acc_thpower",
    "has_cleaning_campaign": "filter_",
}

features_col = [
    "boiler_temp",
    "total_thermal_power_kwh_since_last_cleaning",
    "SCC_temp_degreC",
    "std_thermal_power_kwh_since_last_cleaning",
    "std_solid_waste_flow_kg_since_last_cleaning",
    "std_T105_flow_kg_since_last_cleaning",
    "std_odslime_flow_kg_since_last_cleaning",
    "std_ew_flow_kg_since_last_cleaning",
    "std_organic_acid_flow_kg_since_last_cleaning",
    "boiler_transfer_W_K_m2",
    "T105_flow_kg_hour",
    "ew_flow_kg_hour",
    "std_ods_flow_kg_since_last_cleaning",
    "time_from_last_campaign",
    "average_SCC_temp_degreC_last_cleaning_campaign",
    "total_solid_waste_flow_kg_since_last_cleaning",
    "last_cleaning_campaign_duration",
]

features_col_final = [
    "boiler_temp",
    "total_thermal_power_kwh_since_last_cleaning",
    "std_thermal_power_kwh_since_last_cleaning",
    "std_solid_waste_flow_kg_since_last_cleaning",
    "std_T105_flow_kg_since_last_cleaning",
    "SCC_temp_degreC",
    "std_odslime_flow_kg_since_last_cleaning",
    "std_ew_flow_kg_since_last_cleaning",
    "std_organic_acid_flow_kg_since_last_cleaning",
    "boiler_transfer_W_K_m2",
    "total_T105_flow_kg_since_last_cleaning",
    "ew_flow_kg_hour",
    "std_ods_flow_kg_since_last_cleaning",
    "Aqueous_flow_kg_hour",
    "time_from_last_campaign",
    "average_SCC_temp_degreC_last_cleaning_campaign",
    "total_solid_waste_flow_kg_since_last_cleaning",
    "last_cleaning_campaign_duration",
]

features_col = list(map(lambda x: rename_map.get(x, x), features_col))


SAMPL_RT = 6
OUTSTEPS = int(21 * 24 / SAMPL_RT)
INSTEPS = int(7 * 24 / SAMPL_RT)
num_features = len(features_col)
num_labels = 1


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


# features_col = ['SCC_temp_cleaning',
# 'uptime',
# 'cleaning_time',
# 'acc_T105',
# 'acc_odslime',
# 'acc_solidwaste',
# 'acc_ew',
# 'acc_specialflow',
# 'acc_organicacid',
# 'acc_ods']


window_128 = window_from_data(
    features_col=features_col,
    label_columns=["boiler_transfer"],
    reload_data=True,
    batch_size=128,
)
window_32 = window_from_data(
    features_col=features_col,
    label_columns=["boiler_transfer"],
    reload_data=False,
    batch_size=32,
)


windows = {32: window_32, 128: window_128}

HP_CELLS_MULTIPL = hp.HParam("x_cells", hp.Discrete([3, 7, 11, 15]))

HP_BATCH = hp.HParam("batch_size", hp.Discrete([32, 128]))


MAE_scaled = "Average scaled MAE"
MAE_percent_scaled = "Average scaled MAE %"


def train_test_model(hparams):

    optimizer = tf.optimizers.Adam()
    callback_mod = "EarlyStopping"
    loss_mod = "mae"
    model_generator = lstm_generator(hparams[HP_CELLS_MULTIPL])

    model_name = "{}_batch_{}xcells".format(
        hparams[HP_BATCH], hparams[HP_CELLS_MULTIPL]
    )

    val_performance, _, cli = training_model(
        model_generator=model_generator,
        window_=windows[hparams[HP_BATCH]],
        model_name=model_name,
        MAX_EPOCHS=2500,
        loss_mod=loss_mod,
        optimizer=optimizer,
        plot=False,
    )

    # save_model
    cli.model.save("experimental_models/transfer_{}".format(model_name))
    return val_performance


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        val_performance = train_test_model(hparams)

        mae_scaled = val_performance[1]
        mae_percent_scaled = val_performance[2]

        tf.summary.scalar(MAE_scaled, mae_scaled, step=1)
        tf.summary.scalar(MAE_percent_scaled, mae_percent_scaled, step=1)


def tensorboard_gen_mono():

    with tf.summary.create_file_writer("logs/hparam_tuning").as_default():
        hp.hparams_config(
            hparams=[HP_CELLS_MULTIPL, HP_BATCH],
            metrics=[
                hp.Metric(MAE_scaled, display_name="Average scaled MAE"),
                hp.Metric(MAE_percent_scaled, display_name="Average scaled MAE%"),
            ],
        )

    session_num = 0

    for mult_cells in HP_CELLS_MULTIPL.domain.values:
        for batch_size in HP_BATCH.domain.values:
            hparams = {HP_CELLS_MULTIPL: mult_cells, HP_BATCH: batch_size}
            # run_name = "run-%d" % session_num

            run_name = "run {}: {}_batch {}_xcells".format(
                session_num, batch_size, mult_cells
            )

            logger.info("--- Starting trial: %s" % run_name)
            logger.info({h.name: hparams[h] for h in hparams})
            run("logs/hparam_tuning/" + run_name, hparams)
            session_num += 1
