try:
    import tensorflow as tf
    import keras_tuner as kt
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
    import random
    from sklearn.metrics import mean_squared_error
    from os import system
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from Forecaster_Utils import predict_with_padasip_rls, convert_change_in_load_to_base_load
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


class ModelTrainer:
    MODEL_REGISTRY = {}

    def __init__(self, model_name, hyperparameters=None):

        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Model {model_name} is not registered.")
        self.model = self.MODEL_REGISTRY[model_name](hyperparameters)

    @classmethod
    def register_model(self, name):
        def wrapper(cls):
            self.MODEL_REGISTRY[name] = cls
            return cls

        return wrapper

    def print_best_epoch_summary(self, history, metric="loss"):
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

    def train(self, X, y):
        return self.model.train(X, y)


@ModelTrainer.register_model(name="LSTM")
class LSTMTrainer(ModelTrainer):
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

        self.print_best_epoch_summary(history, metric="loss")
        self.print_best_epoch_summary(history, metric="mae")

        return model


@ModelTrainer.register_model(name="ANN")
class ANNTrainer(ModelTrainer):
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

        self.print_best_epoch_summary(history, metric="loss")
        self.print_best_epoch_summary(history, metric="mae")

        return model


@ModelTrainer.register_model(name="LightGBM")
class LightGBMTrainer(ModelTrainer):
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


class ForecasterTool:
    def __init__(self, feeder_id: int, scenario: str, version: str = None, model_arch: str = "LSTM", is_hyperparameter_search: bool = False):
        self.version = version or datetime.now().strftime("%Y%m%d")
        self.feeder_id = feeder_id
        self.scenario = scenario
        self.model_arch = model_arch
        self.is_hyperparameter_search = is_hyperparameter_search
        if self.model_arch == "LSTM":
            if self.is_hyperparameter_search:
                self.tag = f"exp_HP"
            else:
                self.tag = "main"
        else:
            self.tag = f"exp_{self.model_arch}"

        self.train_start_date = None
        self.train_end_date = None
        self.db_client = DatabaseManager(tag=self.tag)
        self.METADATA_SCHEMA = "metadata"
        self.DATA_SCHEMA = "data"
        self.ML_SCHEMA = "ml"
        self.scaler = None
        self.MODELS_STORAGE_BUCKET = "models"
        self.RLS_BUCKET = "rls-combiners"
        self.TRAIN_START_DATE = "2024-01-01 00:00:00+00"
        self.TRAIN_END_DATE = "2024-05-31 23:59:59+00"
        self.VALIDATION_START_DATE = "2024-06-01 00:00:00+00"
        self.VALIDATION_END_DATE = "2024-06-30 23:59:59+00"
        self.DAY_HOURS = list(range(6, 20 + 1))
        self.NIGHT_HOURS = list(range(0, 6)) + list(range(21, 24))
        self.calendar_features = HolidayFeatures(
            calendar=country_holidays(country="BB"),
            return_categorical=True,
            include_weekend=False,
            return_indicator=True,
            include_bridge_days=True,
            return_dummies=False,
        )

    def initialize_rls_filters(self, num_outputs):
        return [FilterRLS(n=2, mu=0.99, w="random") for _ in range(num_outputs)]

    def predict_with_padasip_rls(self, rls_filters, actuals, predictions1, predictions2):
        n_samples, n_outputs = predictions1.shape
        if len(rls_filters) != n_outputs:
            raise ValueError("Number of RLS filters does not match number of prediction outputs.")
        combined_predictions = np.zeros_like(predictions1)
        for t in range(n_samples):
            for k in range(n_outputs):
                x_k = np.array([predictions1[t, k], predictions2[t, k]])
                d_k = actuals[t, k]
                combined_predictions[t, k] = rls_filters[k].predict(x_k)
                rls_filters[k].adapt(d_k, x_k)
        return combined_predictions

    def plot_history(self, history, metric="loss"):
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

    def convert_change_in_load_to_base_load(self, X, y_pred_change):
        prev_day_cols = [col for col in X.columns if col.startswith("Prev_Day_Net_Load_Demand_Hour_")]

        if len(prev_day_cols) != y_pred_change.shape[-1]:
            print(f"Expected {len(prev_day_cols)} columns, but got {y_pred_change.shape[1]} columns in y_pred_change.")
            raise ValueError("Mismatch in number of columns between prev day load and change predictions")
        prev_day_indices = [X.columns.get_loc(col) for col in prev_day_cols]
        prev_day_load = X.iloc[:, prev_day_indices].values
        return prev_day_load + y_pred_change

    def train_base_and_change_models(self, X, y_base, y_change, model_arch="LSTM", hyperparameters=None):
        trainer = ModelTrainer(model_arch, hyperparameters=hyperparameters)
        if not trainer:
            raise ValueError(f"Unsupported model architecture: {model_arch}")

        print(f"Training {model_arch} model...")
        print(f"X shape: {X.shape}, y_base shape: {y_base.shape}, y_change shape: {y_change.shape}")

        base_model = trainer.train(X, y_base)

        if model_arch == "LightGBM":
            change_model = None
        else:
            change_model = trainer.train(X, y_change)

        return base_model, change_model

    def find_clusters(self, df, n_clusters=3, is_train=False, feeder_id=None):
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
    def pivot_df(self, df, value_columns, expected_hours, features=False, is_train=False):
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

            date_df = pd.DataFrame(index=pivoted.index)
            holiday_df = self.calendar_features.fit_transform(date_df)
            holiday_df.index = pivoted.index

            pivoted["is_weekend"] = pivoted.index.dayofweek.isin([5, 6]).astype(int)
            pivoted["is_weekday"] = pivoted.index.dayofweek.isin([0, 1, 2, 3, 4]).astype(int)
            pivoted["is_holiday"] = holiday_df["is_holiday"].astype(int)

        return pivoted

    def detect_outlier_days(self, df, holiday_df):
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

    def get_days_when_prev_day_not_available(self, df):
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
    def feature_engineer(self, df, is_train=False):
        df_processed = df.copy()

        base_target_column = "Net_Load_Demand"
        change_target_column = "Net_Load_Change"

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

        scenario_hours = self.DAY_HOURS if self.scenario == "Day" else self.NIGHT_HOURS if self.scenario == "Night" else list(range(24))

        X = self.pivot_df(df_processed, feature_cols, scenario_hours, features=True, is_train=is_train)
        y_base = self.pivot_df(df_processed, [base_target_column], scenario_hours, features=False, is_train=is_train)
        y_change = self.pivot_df(df_processed, [change_target_column], scenario_hours, features=False, is_train=is_train)

        valid_indices = X.dropna().index
        X = X.loc[valid_indices]
        y_base = y_base.loc[valid_indices]
        y_change = y_change.loc[valid_indices]

        if is_train:
            # Detect outlier days in the base load data
            holiday_df = X[["is_holiday"]]
            outlier_days = self.detect_outlier_days(y_base, holiday_df)
            if len(outlier_days) > 0:
                print("Dropping outlier days from training data.")
                X = X[~X.index.isin(outlier_days)]
                y_base = y_base[~y_base.index.isin(outlier_days)]
                y_change = y_change[~y_change.index.isin(outlier_days)]

            prev_day_missing_dates = self.get_days_when_prev_day_not_available(y_base)
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

    def run_lstm_hyperparameter_search(self, X_train, y_train_base, y_train_change, X_val, y_val_base, y_val_change, max_trials=20):

        X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))

        def build_base_model(hp):
            model = Sequential()
            model.add(Input(shape=(X_train.shape[1:])))
            model.add(LSTM(hp.Choice("lstm_units", [8, 16, 32, 64, 128, 256]), return_sequences=True))
            model.add(Dense(hp.Choice("dense_units", [8, 16, 32, 64, 128, 256]), activation="relu"))
            model.add(Dropout(hp.Choice("dropout_rate", [0.1, 0.2, 0.3, 0.4, 0.5])))

            if hp.Boolean("add_extra_dense_layer"):
                model.add(Dense(hp.Choice("extra_dense_units", [8, 16, 32, 64, 128, 256]), activation="relu"))
                model.add(Dropout(hp.Choice("extra_dense_dropout", [0.1, 0.2, 0.3, 0.4])))

            model.add(Flatten())
            model.add(Dense(y_train_base.shape[-1], activation="linear"))

            model.compile(optimizer=Adam(learning_rate=hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])), loss=Huber(), metrics=["mae"])
            return model

        def build_change_model(hp):
            model = Sequential()
            model.add(Input(shape=(X_train.shape[1:])))
            model.add(LSTM(hp.Choice("lstm_units", [16, 32, 64]), return_sequences=True))
            model.add(Dense(hp.Choice("dense_units", [16, 32, 64, 128]), activation="relu"))
            model.add(Dropout(hp.Choice("dropout_rate", [0.0, 0.1, 0.2])))
            model.add(Flatten())
            model.add(Dense(y_train_change.shape[-1], activation="linear"))

            model.compile(optimizer=Adam(learning_rate=hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])), loss=Huber(), metrics=["mae"])
            return model

        # Tuner for Base Model
        tuner_base = kt.BayesianOptimization(
            build_base_model,
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=1,
            overwrite=True,
            directory="tuner_dir",
            project_name="lstm_hp_search_base",
        )

        tuner_base.search(
            X_train,
            y_train_base,
            epochs=100,
            validation_data=(X_val, y_val_base),
            callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
            verbose=1,
        )

        best_base_model = tuner_base.get_best_models(num_models=1)[0]
        best_base_hyperparams = tuner_base.get_best_hyperparameters(1)[0].values

        print("\nBest Base Model Hyperparameters:", best_base_hyperparams)

        self.db_client.save_model_to_supabase(
            best_base_model,
            feeder_id=self.feeder_id,
            arch_type="LSTM",
            scenario=self.scenario,
            version=self.version,
            start_ts=self.train_start_date,
            end_ts=self.train_end_date,
            hyperparams=best_base_hyperparams,
            load_type="base",
        )

        # Tuner for Change Model
        tuner_change = kt.BayesianOptimization(
            build_change_model,
            objective="val_loss",
            max_trials=max_trials // 2,
            executions_per_trial=1,
            overwrite=True,
            directory="tuner_dir",
            project_name="lstm_hp_search_change",
        )

        tuner_change.search(
            X_train,
            y_train_change,
            epochs=100,
            validation_data=(X_val, y_val_change),
            callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
            verbose=1,
        )

        best_change_model = tuner_change.get_best_models(num_models=1)[0]
        best_change_hyperparams = tuner_change.get_best_hyperparameters(1)[0].values

        print("\nBest Change Model Hyperparameters:", best_change_hyperparams)

        self.db_client.save_model_to_supabase(
            best_change_model,
            feeder_id=self.feeder_id,
            arch_type="LSTM",
            scenario=self.scenario,
            version=self.version,
            start_ts=self.train_start_date,
            end_ts=self.train_end_date,
            hyperparams=best_change_hyperparams,
            load_type="change",
        )

        return best_base_model, best_change_model

    def run_training(self, train_start_date, train_end_date):
        """Orchestrates the training process for different architectures."""
        print(f"\n--- Starting Training Run ---")
        print(f"Feeder: {self.feeder_id}, Arch: {self.model_arch}, Scenario: {self.scenario}, Version: {self.version}")
        print(f"Train Period: {train_start_date} to {train_end_date}")

        # Determine if the architecture uses Keras
        is_keras_model = self.model_arch in ["ANN", "LSTM"]

        if is_keras_model and not KERAS_AVAILABLE:
            print(f"ERROR: Keras is required for architecture '{self.model_arch}' but is not installed.")
            return

        if self.tag != "main":
            print(f"Tag '{self.tag}' is not 'main'. Running training and forecasts in experiment mode.")

        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

        fetch_train_start = (pd.to_datetime(self.train_start_date) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S%z")
        train_df_raw = self.db_client.fetch_data(self.feeder_id, fetch_train_start, train_end_date)

        print(train_df_raw)

        if train_df_raw.empty:
            print("Insufficient raw data fetched. Aborting training.")
            return

        hyperparameters = {}

        self.scaler = Scaler(
            feeder_id=self.feeder_id, scenario_type=self.scenario, train_start_date=self.train_start_date, train_end_date=self.train_end_date
        )
        train_df_scaled = self.scaler.fit_transform(train_df_raw)

        # print(train_df_scaled)
        # sys.exit(1)

        # Perform feature engineering and scaling
        X_train, y_train_base, y_train_change, common_timestamps = self.feature_engineer(train_df_scaled, is_train=True)

        print(f"Common timestamps: {common_timestamps}")

        # system.exit(1)

        print(
            f"Feature engineering complete. Shapes: X_train {X_train.shape}, y_train_base {y_train_base.shape}, y_train_change {y_train_change.shape}"
        )

        print(f"NaN check after feature engineering:")
        print(f"  X_train NaNs: {np.isnan(X_train).any().any()}")
        print(f"  y_train_base NaNs: {np.isnan(y_train_base).any().any()}")
        print(f"  y_train_change NaNs: {np.isnan(y_train_change).any().any()}")

        if self.tag == "exp_hp":
            print("Running hyperparameter tuning for LSTM...")
            val_df_raw = self.db_client.fetch_data(self.feeder_id, "2024-06-01 00:00:00+00", "2024-06-30 23:59:59+00")
            val_df_scaled = self.scaler.transform(val_df_raw)
            X_val, y_val_base, y_val_change, common_timestamps_val = self.feature_engineer(val_df_scaled, is_train=False)

            self.run_lstm_hyperparameter_search(X_train, y_train_base, y_train_change, X_val, y_val_base, y_val_change, max_trials=10)

            return

        base_model, change_model = self.train_base_and_change_models(
            X_train, y_train_base, y_train_change, model_arch=self.model_arch, hyperparameters=hyperparameters
        )

        print(f"Base and change models trained.")

        # Save models to Supabase storage

        self.db_client.save_model_to_supabase(
            base_model,
            feeder_id=self.feeder_id,
            arch_type=self.model_arch,
            scenario=self.scenario,
            version=self.version,
            start_ts=self.train_start_date,
            end_ts=self.train_end_date,
            hyperparams=hyperparameters,
            load_type="base",
        )

        if change_model is not None:
            self.db_client.save_model_to_supabase(
                change_model,
                feeder_id=self.feeder_id,
                arch_type=self.model_arch,
                scenario=self.scenario,
                version=self.version,
                start_ts=self.train_start_date,
                end_ts=self.train_end_date,
                hyperparams=hyperparameters,
                load_type="change",
            )

        print(f"Models saved to Supabase.")

        base_model, change_model = self.db_client.load_models_from_supabase(self.feeder_id, self.scenario, self.model_arch)
        print(f"Models loaded from Supabase.")

        if self.model_arch == "LSTM" or self.model_arch == "ANN":
            base_model.summary()
            if change_model is not None:
                change_model.summary()
            print(f"Model summaries printed.")

        if self.model_arch == "LSTM":
            y_base_pred = base_model.predict(X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1])))
            y_change_pred = change_model.predict(X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1])))
            y_change_pred = self.convert_change_in_load_to_base_load(X_train, y_change_pred)
        elif self.model_arch == "ANN" or self.model_arch == "LightGBM":
            y_base_pred = base_model.predict(X_train.values.reshape((X_train.shape[0], X_train.shape[1])))
            print(f"y_base_pred shape: {y_base_pred.shape}")

            if change_model is not None:
                y_change_pred = change_model.predict(X_train.values.reshape((X_train.shape[0], X_train.shape[1])))
                print(f"y_change_pred shape: {y_change_pred.shape}")
                y_change_pred_flat = pd.DataFrame(y_change_pred.flatten(), index=common_timestamps, columns=["Net_Load_Demand"])
                y_change_pred_flat = self.scaler.inverse_transform(y_change_pred_flat)
                print(f"y_change_pred_flat after inverse transform: {y_change_pred_flat}")

            # print(y_base_pred)

            # scaler_y_base = db_client.load_scaler(feeder_id, version="latest", purpose="target", load_type="base", scenario=scenario)

            # y_base_pred = pd.DataFrame(y_base_pred, index=y_train_base.index, columns=y_train_base.columns)
        y_base_pred_flat = pd.DataFrame(y_base_pred.flatten(), index=common_timestamps, columns=["Net_Load_Demand"])
        y_true_base_flat = pd.DataFrame(y_train_base.values.flatten(), index=common_timestamps, columns=["Net_Load_Demand"])

        y_base_pred_flat = self.scaler.inverse_transform(y_base_pred_flat)
        y_true_base_flat = self.scaler.inverse_transform(y_true_base_flat)

        # print(f"y_change_pred after inverse transform: {y_change_pred}")

        print(f"Predictions made.")
        print(f"Inverse transformed predictions.")
        print(f"y_base_pred_flat after inverse transform: {y_base_pred_flat}")
        print(f"y_true_base_flat after inverse transform: {y_true_base_flat}")

    def forecast_range(self, forecast_start_date, forecast_end_date, save_daily_rls=True, new_rls=True):
        if self.tag != "main":
            print(f"Tag '{self.tag}' is not 'main'. Running training and forecasts in experiment mode.")

        current_date = pd.to_datetime(forecast_start_date).date()
        end_date = pd.to_datetime(forecast_end_date).date()

        all_forecasts_rls = []
        all_forecasts_base = []

        while current_date <= end_date:
            print(f"\nProcessing {self.scenario} forecast for {current_date} (Feeder {self.feeder_id})")
            target_date = current_date + timedelta(days=1)

            try:
                df = self.db_client.fetch_data(self.feeder_id, current_date.isoformat(), (current_date + timedelta(days=1)).isoformat())
                if df.empty:
                    print(f"No data for {current_date}. Skipping.")
                    current_date += timedelta(days=1)
                    continue

                if self.scaler is None:
                    self.scaler = Scaler(
                        feeder_id=self.feeder_id,
                        scenario_type=self.scenario,
                        train_start_date=self.train_start_date,
                        train_end_date=self.train_end_date,
                    )
                df_scaled = self.scaler.transform(df)

                # Engineer features and targets using trainer pipeline
                X, y_base, y_change, common_timestamps = self.feature_engineer(df_scaled, is_train=False)

                if X.empty or y_base.empty or y_change.empty:
                    print(f"Skipping {current_date}: Feature engineering returned empty results.")
                    current_date += timedelta(days=1)
                    continue

                base_model, change_model = self.db_client.load_models_from_supabase(self.feeder_id, self.scenario, arch_type=self.model_arch)

                if self.model_arch == "LSTM":
                    X_numpy = X.values.reshape((1, 1, X.shape[1]))
                elif self.model_arch == "ANN" or self.model_arch == "LightGBM":
                    X_numpy = X.values.reshape((1, X.shape[1]))

                y_pred_base = base_model.predict(X_numpy)

                y_pred_base_true = pd.DataFrame(y_pred_base.flatten(), columns=["Net_Load_Demand"], index=common_timestamps)
                y_pred_base_true = self.scaler.inverse_transform(y_pred_base_true)
                actuals = pd.DataFrame(y_base.values.flatten(), columns=["Net_Load_Demand"], index=common_timestamps)
                actuals = self.scaler.inverse_transform(actuals)

                y_pred_base_true = y_pred_base_true.rename(columns={"Net_Load_Demand": "forecast_value"})
                actuals = actuals.rename(columns={"Net_Load_Demand": "actual_value"})

                if change_model is not None:
                    y_pred_change = change_model.predict(X_numpy) if change_model else np.zeros_like(y_pred_base)
                    y_pred_combined_change_prev_day_base = convert_change_in_load_to_base_load(X, y_pred_change)
                    print("Converted load shape: ", y_pred_combined_change_prev_day_base.shape)

                    # Load or initialize RLS filters
                    rls_filters = self.db_client.load_rls_filters(self.feeder_id, self.version, self.scenario, self.model_arch)
                    if rls_filters is None or new_rls:
                        print("Initializing new RLS filters.")
                        new_rls = False
                        rls_filters = self.initialize_rls_filters(y_pred_base.shape[-1])

                    y_pred_rls_scaled = self.predict_with_padasip_rls(rls_filters, y_base.values, y_pred_base, y_pred_combined_change_prev_day_base)
                    print("RLS prediction: ", y_pred_rls_scaled)
                    print("True: ", y_base)
                    print("Base prediction: ", y_pred_base)
                    print("Change converted prediction: ", y_pred_combined_change_prev_day_base)
                    y_pred_rls_scaled = pd.DataFrame(y_pred_rls_scaled.flatten(), columns=["Net_Load_Demand"], index=common_timestamps)

                    y_pred_rls = self.scaler.inverse_transform(y_pred_rls_scaled)
                    y_pred_rls = y_pred_rls.rename(columns={"Net_Load_Demand": "forecast_value"})
                    y_pred_rls = pd.concat([y_pred_rls, actuals], axis=1)
                    all_forecasts_rls.append(y_pred_rls)

                    if save_daily_rls:
                        self.db_client.save_rls_filters(rls_filters, self.feeder_id, self.version, self.scenario, self.model_arch)

                    self.db_client.save_forecasts(
                        self.feeder_id,
                        self.version,
                        scenario_type=self.scenario,
                        model_architecture_type=self.model_arch,
                        forecasts_df=y_pred_rls,
                    )
                else:
                    self.db_client.save_forecasts(
                        self.feeder_id,
                        self.version,
                        scenario_type=self.scenario,
                        model_architecture_type=self.model_arch,
                        forecasts_df=y_pred_base_true,
                    )

                all_forecasts_base.append(y_pred_base_true)

            except Exception as e:
                print(f"Error on {current_date}: {e}")
                traceback.print_exc()

            current_date += timedelta(days=1)

        # Save to CSV
        # if all_forecasts_rls:
        #     result_df = pd.concat(all_forecasts_rls)
        #     result_df.to_csv(FORECAST_CSV_PATH.format(feeder_id=self.feeder_id, scenario=self.scenario, type="rls"))
        #     print(f"Forecast results saved to {FORECAST_CSV_PATH.format(feeder_id=self.feeder_id, scenario=self.scenario, type='rls')}")

        # if all_forecasts_base:
        #     result_df = pd.concat(all_forecasts_base)
        #     result_df.to_csv(FORECAST_CSV_PATH.format(feeder_id=feeder_id, scenario=scenario, type="base"))
        #     print(f"Forecast results saved to {FORECAST_CSV_PATH.format(feeder_id=feeder_id, scenario=scenario, type='base')}")
        else:
            print("No forecasts generated.")
