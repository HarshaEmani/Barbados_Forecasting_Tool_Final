from os import system


try:
    import tensorflow as tf
    from keras.models import Sequential, save_model, load_model
    from keras.layers import Dense, LSTM, Input, Dropout, Lambda, Layer, Flatten
    from DB_Utils import NormalizeLayer
    from keras.callbacks import EarlyStopping
    from padasip.filters import FilterRLS
    from sklearn.multioutput import MultiOutputRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from Scaler import Scaler
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta, timezone
    import json
    import pickle
    import os
    import sys
    import argparse
    import keras
    from keras.optimizers import Adam
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping
    import keras.backend as K
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.losses import Huber
    from Scaler_Manager import ScalerManager
    from tslearn.clustering import TimeSeriesKMeans
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from Forecaster_Utils import predict_with_padasip_rls, convert_change_in_load_to_base_load

    # from DB_Utils import (
    #     fetch_data,
    #     save_pickle_artifact,
    #     log_model_metadata,
    #     store_validation_results,
    #     select_model_for_forecast,
    #     load_artifact_from_storage,
    # )
    import tempfile
    from DB_Manager import DatabaseManager
    from supabase import create_client, Client
    import traceback
    import plotly.express as px
    from dotenv import load_dotenv, find_dotenv
    import tensorflow as tf
    from sktime.transformations.series.holiday import HolidayFeatures
    from sktime.transformations.series.date import DateTimeFeatures
    from holidays import country_holidays
    import matplotlib.pyplot as plt

    # from tensorflow.keras.layers import

    KERAS_AVAILABLE = True
    PADASIP_AVAILABLE = True

    np.set_printoptions(suppress=True)

    load_dotenv()
    print("Env file found at location: ", find_dotenv())

except ImportError:
    print("TensorFlow/Keras not found. Keras models cannot be trained/saved natively.")
    KERAS_AVAILABLE = False
    # Define dummy classes if needed for type checking, though not strictly necessary here
    Sequential, save_model, load_model = object, lambda x, y: None, lambda x: None
    Dense, LSTM, Input, Dropout, EarlyStopping = object, object, object, object, object

    print("padasip not found. RLS filters cannot be loaded/used.")
    PADASIP_AVAILABLE = False
    FilterRLS = object


# --- Configuration & Constants --- (Same as before)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SECRET_KEY")
if not SUPABASE_URL or not SUPABASE_KEY or "YOUR_SUPABASE_URL" in SUPABASE_URL:
    print("Error: SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
    # sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
DATA_SCHEMA = "data"
ML_SCHEMA = "ml"
METADATA_SCHEMA = "metadata"
STORAGE_BUCKET = "models"
FEEDER_ID_TO_TRAIN = 1
SCENARIO = "Day"  # Example
MODEL_VERSION = "v1.1_Is_Weekend"  # Updated version
TRAIN_START_DATE = "2024-01-01 00:00:00+00"
TRAIN_END_DATE = "2024-05-31 23:59:59+00"
VALIDATION_START_DATE = "2024-06-01 00:00:00+00"
VALIDATION_END_DATE = "2024-06-30 23:59:59+00"
DAY_HOURS = list(range(6, 20 + 1))
NIGHT_HOURS = list(range(0, 6)) + list(range(21, 24))
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
TEMP_DIR = os.path.join(script_dir, "tmp")


def plot_history(history, metric="loss"):
    """Plot training history."""
    if history is None:
        print("No history to plot.")
        return

    if metric not in history.history:
        print(f"Metric '{metric}' not found in history.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(history.history[metric], label="Train")
    plt.plot(history.history[f"val_{metric}"], label="Validation")
    plt.title(f"Model {metric.capitalize()}")
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()


def convert_change_in_load_to_base_load(X, y_pred_change):
    prev_day_cols = [col for col in X.columns if col.startswith("Prev_Day_Net_Load_Demand_Hour_")]

    if len(prev_day_cols) != y_pred_change.shape[-1]:
        print(f"Expected {len(prev_day_cols)} columns, but got {y_pred_change.shape[1]} columns in y_pred_change.")
        raise ValueError("Mismatch in number of columns between prev day load and change predictions")
    prev_day_indices = [X.columns.get_loc(col) for col in prev_day_cols]
    prev_day_load = X.iloc[:, prev_day_indices].values
    return prev_day_load + y_pred_change


def print_best_epoch_summary(history, metric="loss"):
    """Print the epoch with the best (lowest) validation metric."""
    if history is None:
        print("No training history available.")
        return

    val_metric_key = f"val_{metric}"
    if val_metric_key not in history.history:
        print(f"No validation metric '{val_metric_key}' found in history.")
        return

    best_epoch = int(np.argmin(history.history[val_metric_key]))
    best_train = history.history[metric][best_epoch]
    best_val = history.history[val_metric_key][best_epoch]

    print(f"Best Epoch: {best_epoch + 1}")
    print(f" - Train {metric}: {best_train:.4f}")
    print(f" - Val {metric}:   {best_val:.4f}")


# ---------------- Model Trainer Registry ---------------- #
MODEL_REGISTRY = {}


def register_model(name):
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return wrapper


@register_model("LSTM")
class LSTMTrainer:
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters

    def build_model(self, X, y):
        model = Sequential()
        model.add(Input(shape=(X.shape[1:])))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dense(256, activation="relu", kernel_regularizer=l2(1e-4)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        # model.add(Dropout(0.3))
        model.add(Dense(y.shape[-1], activation="linear"))  # Linear for regression
        return model

    def train(self, X, y):
        X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))

        model = self.build_model(X_lstm, y)
        lr = self.hyperparameters.get("learning_rate", 0.001) if self.hyperparameters else 0.001

        model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(), metrics=["mae"])  # <<-- Huber loss here

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=25, min_lr=1e-6, verbose=1),
        ]

        history = model.fit(
            X_lstm,
            y,
            epochs=250,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
            validation_split=0.05,
            shuffle=False,
        )

        print_best_epoch_summary(history, metric="loss")
        print_best_epoch_summary(history, metric="mae")

        return model


@register_model("ANN")
class ANNTrainer:
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters

    def build_model(self, X, y):
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        model = Sequential()
        model.add(Input(shape=(X.shape[1:])))
        model.add(Dense(256, activation="relu", kernel_regularizer=l2(1e-4)))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(y.shape[-1], activation="linear"))  # Linear for regression
        return model

    def train(self, X, y):
        # X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))

        model = self.build_model(X, y)
        lr = self.hyperparameters.get("learning_rate", 0.001) if self.hyperparameters else 0.001

        model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(), metrics=["mae"])  # <<-- Huber loss here

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=80, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=55, min_lr=1e-6, verbose=1),
        ]

        history = model.fit(
            X,
            y,
            epochs=250,
            batch_size=4,
            callbacks=callbacks,
            verbose=0,
            validation_split=0.05,
            shuffle=False,
        )

        print_best_epoch_summary(history, metric="loss")
        print_best_epoch_summary(history, metric="mae")

        return model


@register_model("LightGBM")
class LightGBMTrainer:
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters or {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
        }

    def train(self, X, y):
        model = MultiOutputRegressor(LGBMRegressor(**self.hyperparameters, verbosity=-1))
        model.fit(X, y)
        return model


def train_base_and_change_models(X, y_base, y_change, model_arch="LSTM", hyperparameters=None):
    trainer_class = MODEL_REGISTRY.get(model_arch)
    if not trainer_class:
        raise ValueError(f"Unsupported model architecture: {model_arch}")

    print(f"Training {model_arch} model...")
    print(f"X shape: {X.shape}, y_base shape: {y_base.shape}, y_change shape: {y_change.shape}")

    trainer = trainer_class(hyperparameters)
    base_model = trainer.train(X, y_base)

    if model_arch == "LightGBM":
        change_model = None
    else:
        change_model = trainer.train(X, y_change)

    return base_model, change_model


def find_clusters(df, n_clusters=3, is_train=False, feeder_id=None):
    """Find clusters in the data using KMeans clustering."""

    if is_train:
        kmeans = TimeSeriesKMeans(n_clusters=n_clusters, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cluster_labels = kmeans.fit_predict(df.values)
        rf.fit(df.values, cluster_labels)

        with open(f"rf_model_feeder_{feeder_id}.pkl", "wb") as f:
            pickle.dump(rf, f)
    else:
        with open(f"rf_model_feeder_{feeder_id}.pkl", "rb") as f:
            rf = pickle.load(f)

    clusters = rf.predict(df.values)

    return clusters


# --- Pivot Helper Function ---
def pivot_df(df, value_columns, expected_hours, features=False, is_train=False):
    df = df.copy()
    df["date"] = df.index.date
    df["hour"] = df.index.hour

    pivoted = df.pivot_table(index="date", columns="hour", values=value_columns)

    if isinstance(value_columns, list) and len(value_columns) > 1:
        pivoted.columns = [f"{col[0]}_Hour_{col[1]}" for col in pivoted.columns]
    else:
        col_name = value_columns[0] if isinstance(value_columns, list) else value_columns
        pivoted.columns = [f"{col_name}_Hour_{col[1]}" for col in pivoted.columns]

    pivoted = pivoted.reindex(columns=[f"{col}_Hour_{hr}" for col in value_columns for hr in expected_hours])

    pivoted.index = pd.to_datetime(pivoted.index)
    if features:
        # pivoted["day_of_week"] = pivoted.index.dayofweek
        # dow_dummies = pd.get_dummies(pivoted["day_of_week"], prefix="dow", dtype=int)

        # Ensure all 7 day columns exist
        # expected_dow_cols = [f"dow_{i}" for i in range(7)]
        # dow_dummies = dow_dummies.reindex(columns=expected_dow_cols, fill_value=0)

        # pivoted = pivoted.drop(columns=["day_of_week"])
        # pivoted = pd.concat([pivoted, dow_dummies], axis=1)

        # Add is_holiday using sktime
        calendar_features = HolidayFeatures(
            calendar=country_holidays(country="BB"),  # Change to your country code
            return_categorical=True,
            include_weekend=False,
            return_indicator=True,
            include_bridge_days=True,
            return_dummies=False,
        )

        date_df = pd.DataFrame(index=pivoted.index)
        holiday_df = calendar_features.fit_transform(date_df)
        holiday_df.index = pivoted.index

        # print(f"Holiday features:\n{holiday_df.loc['2024-03-15':'2024-04-15']}")

        pivoted["is_weekend"] = pivoted.index.dayofweek.isin([5, 6]).astype(int)
        pivoted["is_weekday"] = pivoted.index.dayofweek.isin([0, 1, 2, 3, 4]).astype(int)
        pivoted["is_holiday"] = holiday_df["is_holiday"].astype(int)
    # else:

    return pivoted


def detect_outlier_days(df, holiday_df):
    outlier_df = df.copy()
    outlier_df["is_holiday"] = holiday_df["is_holiday"].astype(int)

    outlier_model = IsolationForest(contamination=0.01, random_state=42)
    outlier_model.fit(outlier_df)
    outlier_labels = outlier_model.predict(outlier_df)
    outlier_df["Outlier"] = outlier_labels

    outlier_days = outlier_df[(outlier_df["is_holiday"] == 0) & (outlier_df["Outlier"] == -1)].index

    if len(outlier_days) == 0:
        print("No outlier days detected.")
    else:
        print(f"Outlier days detected:\n {outlier_days}")

    return outlier_days


def get_days_when_prev_day_not_available(df):
    df = df.copy()
    df["Prev_Day"] = df.index - pd.DateOffset(days=1)
    df["Prev_Day_Available"] = df["Prev_Day"].isin(df.index).astype(int)
    df.loc[df.index[0], "Prev_Day_Available"] = 1

    # df.drop(columns=["Prev_Day", "Prev_Day_Available"], inplace=True)

    prev_day_missing_dates = df[df["Prev_Day_Available"] == 0].index

    if len(prev_day_missing_dates) > 0:
        print(f"Days with missing previous day data:\n {prev_day_missing_dates}")
    else:
        print("No days with missing previous day data.")

    return prev_day_missing_dates


# --- Updated Feature Engineer ---
def feature_engineer(df, feeder_id, scenario, version, is_train=False):
    df_processed = df.copy()

    base_target_column = "Net_Load_Demand"
    change_target_column = "Net_Load_Change"

    # df_processed[change_target_column] = df_processed[base_target_column].diff(24).fillna(0)
    df_processed["Prev_Day_Net_Load_Demand"] = df_processed[base_target_column].shift(24)
    df_processed["Prev_Day_Temperature_Historic"] = df_processed["Temperature_Historic"].shift(24)
    df_processed["Prev_Day_Shortwave_Radiation_Historic"] = df_processed["Shortwave_Radiation_Historic"].shift(24)
    df_processed["Prev_Day_Cloud_Cover_Historic"] = df_processed["Cloud_Cover_Historic"].shift(24)
    df_processed.dropna(inplace=True)

    feature_cols = [
        "Prev_Day_Net_Load_Demand",
        "Prev_Day_Temperature_Historic",
        "Temperature_Forecast",
        "Prev_Day_Shortwave_Radiation_Historic",
        "Shortwave_Radiation_Forecast",
        "Prev_Day_Cloud_Cover_Historic",
        "Cloud_Cover_Forecast",
    ]

    features_scaled = df_processed[feature_cols]
    y_base_scaled = df_processed[[base_target_column]]

    y_change_scaled = y_base_scaled[base_target_column].copy() - features_scaled["Prev_Day_Net_Load_Demand"].copy()
    y_change_scaled.name = change_target_column

    df_processed = pd.concat([features_scaled, y_base_scaled, y_change_scaled], axis=1)
    df_processed.dropna(inplace=True)
    df_processed.to_csv("df_processed.csv")

    scenario_hours = DAY_HOURS if scenario == "Day" else NIGHT_HOURS if scenario == "Night" else list(range(24))

    X = pivot_df(df_processed, feature_cols, scenario_hours, features=True, is_train=is_train)
    y_base = pivot_df(df_processed, [base_target_column], scenario_hours, features=False, is_train=is_train)
    y_change = pivot_df(df_processed, [change_target_column], scenario_hours, features=False, is_train=is_train)

    valid_indices = X.dropna().index
    X = X.loc[valid_indices]
    y_base = y_base.loc[valid_indices]
    y_change = y_change.loc[valid_indices]

    if is_train:
        # Detect outlier days in the base load data
        holiday_df = X[["is_holiday"]]
        outlier_days = detect_outlier_days(y_base, holiday_df)
        if len(outlier_days) > 0:
            print("Dropping outlier days from training data.")
            X = X[~X.index.isin(outlier_days)]
            y_base = y_base[~y_base.index.isin(outlier_days)]
            y_change = y_change[~y_change.index.isin(outlier_days)]

        prev_day_missing_dates = get_days_when_prev_day_not_available(y_base)
        if len(prev_day_missing_dates) > 0:
            print("Dropping days with missing previous day data from training data.")
            X = X[~X.index.isin(prev_day_missing_dates)]
            y_base = y_base[~y_base.index.isin(prev_day_missing_dates)]
            y_change = y_change[~y_change.index.isin(prev_day_missing_dates)]

    # clusters = find_clusters(X, n_clusters=3, is_train=is_train, feeder_id=feeder_id)
    # X["Cluster"] = clusters

    common_timestamps = df_processed.index[np.isin(df_processed.index.date, y_base.index.date)]

    print(f"Reshaped data: X shape {X.shape}, y_base shape {y_base.shape}, y_change shape {y_change.shape}")
    return X, y_base, y_change, common_timestamps


def run_training(feeder_id, model_arch, scenario, version, train_start, train_end, tag="main"):
    """Orchestrates the training process for different architectures."""
    print(f"\n--- Starting Training Run ---")
    print(f"Feeder: {feeder_id}, Arch: {model_arch}, Scenario: {scenario}, Version: {version}")
    print(f"Train Period: {train_start} to {train_end}")

    # Determine if the architecture uses Keras
    is_keras_model = model_arch in ["ANN", "LSTM"]

    if is_keras_model and not KERAS_AVAILABLE:
        print(f"ERROR: Keras is required for architecture '{model_arch}' but is not installed.")
        return

    if tag != "main":
        print(f"Tag '{tag}' is not 'main'. Running training and forecasts in experiment mode.")

    db = DatabaseManager(tag=tag)

    fetch_train_start = (pd.to_datetime(train_start) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S%z")
    train_df_raw = db.fetch_data(feeder_id, fetch_train_start, train_end)

    if train_df_raw.empty:
        print("Insufficient raw data fetched. Aborting training.")
        return

    hyperparameters = {}

    scaler = Scaler(feeder_id=feeder_id, scenario_type=scenario, train_start_date=train_start, train_end_date=train_end)
    train_df_scaled = scaler.transform(train_df_raw)

    # Perform feature engineering and scaling
    X_train, y_train_base, y_train_change, common_timestamps = feature_engineer(train_df_scaled, feeder_id, scenario, version, is_train=True)

    print(f"Common timestamps: {common_timestamps}")

    # system.exit(1)

    print(f"Feature engineering complete. Shapes: X_train {X_train.shape}, y_train_base {y_train_base.shape}, y_train_change {y_train_change.shape}")

    print(f"NaN check after feature engineering:")
    print(f"  X_train NaNs: {np.isnan(X_train).any().any()}")
    print(f"  y_train_base NaNs: {np.isnan(y_train_base).any().any()}")
    print(f"  y_train_change NaNs: {np.isnan(y_train_change).any().any()}")

    base_model, change_model = train_base_and_change_models(
        X_train, y_train_base, y_train_change, model_arch=model_arch, hyperparameters=hyperparameters
    )

    print(f"Base and change models trained.")

    # Save models to Supabase storage

    db.save_model_to_supabase(
        base_model,
        feeder_id=feeder_id,
        arch_type=model_arch,
        scenario=scenario,
        version=version,
        start_ts=train_start,
        end_ts=train_end,
        hyperparams=hyperparameters,
        load_type="base",
    )

    if change_model is not None:
        db.save_model_to_supabase(
            change_model,
            feeder_id=feeder_id,
            arch_type=model_arch,
            scenario=scenario,
            version=version,
            start_ts=train_start,
            end_ts=train_end,
            hyperparams=hyperparameters,
            load_type="change",
        )

    print(f"Models saved to Supabase.")

    base_model, change_model = db.load_models_from_supabase(feeder_id, scenario, model_arch)
    print(f"Models loaded from Supabase.")

    if model_arch == "LSTM" or model_arch == "ANN":
        base_model.summary()
        if change_model is not None:
            change_model.summary()
        print(f"Model summaries printed.")

    if model_arch == "LSTM":
        y_base_pred = base_model.predict(X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1])))
        y_change_pred = change_model.predict(X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1])))
        y_change_pred = convert_change_in_load_to_base_load(X_train, y_change_pred)
    elif model_arch == "ANN" or model_arch == "LightGBM":
        y_base_pred = base_model.predict(X_train.values.reshape((X_train.shape[0], X_train.shape[1])))
        print(f"y_base_pred shape: {y_base_pred.shape}")

        if change_model is not None:
            y_change_pred = change_model.predict(X_train.values.reshape((X_train.shape[0], X_train.shape[1])))
            print(f"y_change_pred shape: {y_change_pred.shape}")
            y_change_pred_flat = pd.DataFrame(y_change_pred.flatten(), index=common_timestamps, columns=["Net_Load_Demand"])
            y_change_pred_flat = scaler.inverse_transform(y_change_pred_flat)
            print(f"y_change_pred_flat after inverse transform: {y_change_pred_flat}")

        # print(y_base_pred)

        # scaler_y_base = db.load_scaler(feeder_id, version="latest", purpose="target", load_type="base", scenario=scenario)

        # y_base_pred = pd.DataFrame(y_base_pred, index=y_train_base.index, columns=y_train_base.columns)
    y_base_pred_flat = pd.DataFrame(y_base_pred.flatten(), index=common_timestamps, columns=["Net_Load_Demand"])
    y_true_base_flat = pd.DataFrame(y_train_base.values.flatten(), index=common_timestamps, columns=["Net_Load_Demand"])

    y_base_pred_flat = scaler.inverse_transform(y_base_pred_flat)
    y_true_base_flat = scaler.inverse_transform(y_true_base_flat)

    # print(f"y_change_pred after inverse transform: {y_change_pred}")

    print(f"Predictions made.")
    print(f"Inverse transformed predictions.")
    print(f"y_base_pred_flat after inverse transform: {y_base_pred_flat}")
    print(f"y_true_base_flat after inverse transform: {y_true_base_flat}")

    # timestamps = np.isin(np.unique(train_df_raw.index.date), y_base_pred.index.date)

    # results = pd.DataFrame(
    #     {
    #         "y_base_pred": y_base_pred,
    #         "y_true": y_train_base,
    #     },
    #     index=train_df_raw.index[24:],
    # )

    # print(f"Results DataFrame created.")
    # print(results)

    # px.line(results, x=results.index, y=["y_base_pred", "y_true"], title="Base Load Predictions vs Actuals").show()

    # except Exception as e:
    #     print(f"Error during feature engineering: {e}")
    #     return
