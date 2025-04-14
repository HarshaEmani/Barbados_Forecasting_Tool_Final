try:
    import tensorflow as tf
    from keras.models import Sequential, save_model, load_model
    from keras.layers import Dense, LSTM, Input, Dropout, Lambda, Layer
    from DB_Utils import NormalizeLayer
    from keras.callbacks import EarlyStopping
    from padasip.filters import FilterRLS
    from sklearn.multioutput import MultiOutputRegressor
    from lightgbm import LGBMRegressor
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
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from Forecaster_Utils import predict_with_padasip_rls, convert_change_in_load_to_base_load
    from DB_Utils import (
        fetch_data,
        save_pickle_artifact,
        log_model_metadata,
        store_validation_results,
        select_model_for_forecast,
        load_artifact_from_storage,
    )
    from supabase import create_client, Client
    import traceback
    import plotly.express as px
    from dotenv import load_dotenv, find_dotenv
    import tensorflow as tf

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

# --- Define Architectures and Scenarios to Run ---
# List all model architectures defined in your run_training function
ARCHITECTURES_TO_RUN = [
    "LightGBM_Baseline",
    "ANN_Baseload",
    "ANN_Change_in_Load",
    "LSTM_Baseload",
    "LSTM_Change_in_Load",
    "ANN_RLS_Combined",
    "LSTM_RLS_Combined",
    "Final_RLS_Combined",
]

SCENARIOS_TO_RUN = ["24hr", "Day", "Night"]


# --- Data Preparation Functions --- (prepare_daily_vectors, feature_engineer_and_scale - unchanged)
def prepare_daily_vectors(df, feature_cols, target_col_list, scenario_hours):  # Expect list
    """Pivots hourly data into daily vectors (one row per day)."""
    print("Pivoting data into daily vectors...")
    df_copy = df.copy()
    df_copy["date"] = df_copy.index.date
    df_copy["hour"] = df_copy.index.hour
    if scenario_hours:
        df_copy = df_copy[df_copy["hour"].isin(scenario_hours)]
    pivoted_X = df_copy.pivot_table(index="date", columns="hour", values=feature_cols)
    pivoted_X.columns = [f"{col[0]}_Hour_{col[1]}" for col in pivoted_X.columns]
    pivoted_y = df_copy.pivot_table(index="date", columns="hour", values=target_col_list)
    if len(target_col_list) == 1:
        if isinstance(pivoted_y, pd.Series):
            pivoted_y = pivoted_y.to_frame()
        pivoted_y.columns = [f"{target_col_list[0]}_Hour_{col[1]}" for col in pivoted_y.columns]  # Adjusted column naming
    else:
        pivoted_y.columns = [f"{col[0]}_Hour_{col[1]}" for col in pivoted_y.columns]
    expected_hours = scenario_hours if scenario_hours else list(range(24))
    ordered_X_columns = [f"{feat}_Hour_{hr}" for feat in feature_cols for hr in expected_hours]
    ordered_y_columns = [f"{tgt}_Hour_{hr}" for tgt in target_col_list for hr in expected_hours]
    pivoted_X = pivoted_X.reindex(columns=ordered_X_columns)
    pivoted_y = pivoted_y.reindex(columns=ordered_y_columns)
    pivoted_X.index = pd.to_datetime(pivoted_X.index)
    pivoted_X["Day_Of_Week"] = pivoted_X.index.dayofweek
    pivoted_X["Is_Holiday"] = 0

    pivoted_X["Is_Weekend"] = pivoted_X["Day_Of_Week"].apply(lambda x: 1 if x in [5, 6] else 0)  # Saturday=5, Sunday=6

    pivoted_X = pd.get_dummies(pivoted_X, columns=["Day_Of_Week"], prefix="DOW", dtype="int")
    original_days = len(pivoted_X)
    valid_indices = pivoted_X.dropna().index
    pivoted_X = pivoted_X.loc[valid_indices]
    pivoted_y = pivoted_y.loc[valid_indices]
    if len(pivoted_X) < original_days:
        print(f"Warning: Dropped {original_days - len(pivoted_X)} days due to missing data after pivoting/feature eng.")
    print(f"Reshaped data: X shape {pivoted_X.shape}, y shape {pivoted_y.shape}")
    return pivoted_X, pivoted_y


def feature_engineer_and_scale(df, scenario, x_scaler=None, y_scaler=None, change_in_load=False, apply_scaling=True):
    """Prepares features, reshapes data, applies MinMaxScaler to X and y."""
    print(f"Starting feature engineering for scenario: {scenario}...")
    df_processed = df.copy()
    df_processed["Net_Load_Change"] = df_processed["Net_Load_Demand"].diff(24).fillna(0)
    df_processed["Prev_Day_Net_Load_Demand"] = df_processed["Net_Load_Demand"].shift(24)
    df_processed["Prev_Day_Temperature_Historic"] = df_processed["Temperature_Historic"].shift(24)
    df_processed["Prev_Day_Shortwave_Radiation_Historic"] = df_processed["Shortwave_Radiation_Historic"].shift(24)
    df_processed["Prev_Day_Cloud_Cover_Historic"] = df_processed["Cloud_Cover_Historic"].shift(24)
    # df_processed.rename(columns={'temperature_2m_forecast': 'Temperature_Forecast', 'shortwave_radiation_forecast': 'Shortwave_Radiation_Forecast', 'cloud_cover_forecast': 'Cloud_Cover_Forecast'}, inplace=True)
    feature_cols = [
        "Prev_Day_Net_Load_Demand",
        "Prev_Day_Temperature_Historic",
        "Temperature_Forecast",
        "Prev_Day_Shortwave_Radiation_Historic",
        "Shortwave_Radiation_Forecast",
        "Prev_Day_Cloud_Cover_Historic",
        "Cloud_Cover_Forecast",
    ]
    target_col = ["Net_Load_Change"] if change_in_load else ["Net_Load_Demand"]
    df_processed = df_processed.dropna(subset=feature_cols + target_col)
    scenario_hours = DAY_HOURS if scenario == "Day" else NIGHT_HOURS if scenario == "Night" else None
    X, y = prepare_daily_vectors(df_processed, feature_cols, target_col, scenario_hours)

    # print(X, y)

    # print("++++++++++++++++++++++++++++++")
    # print(X)
    # print("+++++++++++++++++++++++++++++++++++")
    if X.empty or y.empty:
        print("Warning: No data left after feature engineering and reshaping.")
        return X, y, None, None
    if not apply_scaling:
        print("Scaling is disabled. Returning original data without scaling.")
        return X, y, None, None
    is_training = (x_scaler is None) and (y_scaler is None)
    if is_training:
        print("Fitting new MinMaxScaler on training data (X and y)...")
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_scaled = x_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y)
        print("Scalers fitted.")
        X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        y_scaled_df = pd.DataFrame(y_scaled, index=y.index, columns=y.columns)
        return X_scaled_df, y_scaled_df, x_scaler, y_scaler
    else:
        if x_scaler is None or y_scaler is None:
            raise ValueError("Both x_scaler and y_scaler must be provided for non-training mode.")
        print("Transforming data using provided scalers (X and y)...")
        if not (hasattr(x_scaler, "transform") and hasattr(y_scaler, "transform")):
            raise ValueError("Provided scaler objects must have a 'transform' method.")
        try:
            if hasattr(x_scaler, "feature_names_in_") and list(X.columns) != list(x_scaler.feature_names_in_):
                raise ValueError("Input features mismatch between data and X scaler.")
            elif hasattr(x_scaler, "n_features_in_") and X.shape[1] != x_scaler.n_features_in_:
                raise ValueError(f"Input feature count mismatch: data has {X.shape[1]}, X scaler expects {x_scaler.n_features_in_}")
            if hasattr(y_scaler, "n_features_in_") and y.shape[1] != y_scaler.n_features_in_:
                raise ValueError(f"Target feature count mismatch: data has {y.shape[1]}, y scaler expects {y_scaler.n_features_in_}")
            X_scaled = x_scaler.transform(X)
            y_scaled = y_scaler.transform(y)
            print("Data transformed.")
            X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
            y_scaled_df = pd.DataFrame(y_scaled, index=y.index, columns=y.columns)
            return X_scaled_df, y_scaled_df, None, None
        except Exception as e:
            print(f"Error applying scaler transform: {e}")
            traceback.print_exc()
            raise


# --- Model Training Functions --- (train_lightgbm_model, train_ann_model, train_lstm_model - unchanged)
# --- RLS Functions --- (convert_change_in_load_to_base_load, train_padasip_rls_combiner, predict_with_padasip_rls, run_rls_combination_stage - unchanged)
# --- [Include the definitions for these functions from your previous script here] ---
# Placeholder comment - make sure these functions are defined above run_training
def train_lightgbm_model(X_train_scaled, y_train_scaled, X_val_scaled, y_val_original, y_scaler, verbose=-1):
    """Trains LightGBM model."""
    print(f"Training LightGBM model...")
    lgbm_estimator = LGBMRegressor(n_jobs=-1, random_state=42, verbose=verbose)
    model = MultiOutputRegressor(lgbm_estimator, n_jobs=-1)
    print("Fitting MultiOutputRegressor with LGBM on scaled data...")
    model.fit(X_train_scaled, y_train_scaled)
    print("Model fitting complete.")
    print("Predicting on scaled validation set...")
    y_pred_val_scaled = model.predict(X_val_scaled)
    print("Inverse transforming predictions to original scale...")
    try:
        if y_pred_val_scaled.shape[1] != y_scaler.n_features_in_:
            raise ValueError(
                f"Prediction shape mismatch: predicted {y_pred_val_scaled.shape[1]} features, y_scaler expects {y_scaler.n_features_in_}"
            )
        y_pred_val_original = y_scaler.inverse_transform(y_pred_val_scaled)
    except Exception as e:
        print(f"Error during inverse transform: {e}")
        traceback.print_exc()
        return model, {"mae": np.nan, "rmse": np.nan, "smape": np.nan}, None  # Return None for preds
    print("Calculating validation metrics on original scale...")
    y_val_np = y_val_original.values if isinstance(y_val_original, pd.DataFrame) else y_val_original
    y_pred_val_np = y_pred_val_original
    try:
        mae = mean_absolute_error(y_val_np, y_pred_val_np)
        rmse = np.sqrt(mean_squared_error(y_val_np, y_pred_val_np))
        denominator = np.abs(y_val_np) + np.abs(y_pred_val_np)
        safe_denominator = np.where(denominator == 0, 1, denominator)
        smape_values = 200 * np.abs(y_pred_val_np - y_val_np) / safe_denominator
        smape = np.mean(smape_values)
        validation_metrics = {"mae": mae, "rmse": rmse, "smape": smape}
    except ValueError as metric_err:
        print(f"ERROR calculating metrics: {metric_err}")
        traceback.print_exc()
        validation_metrics = {"mae": np.nan, "rmse": np.nan, "smape": np.nan}
    print("LightGBM training complete.")
    return model, validation_metrics, y_pred_val_original  # Return preds


# def convert_change_in_load_to_base_load(X_original, y_pred_change_original):
#     """Converts predicted change_in_load back to base_load."""
#     X_original_np = X_original.values if isinstance(X_original, pd.DataFrame) else X_original
#     y_pred_change_np = y_pred_change_original.values if isinstance(y_pred_change_original, pd.DataFrame) else y_pred_change_original
#     prev_day_cols = [col for col in X_original.columns if col.startswith("Prev_Day_Net_Load_Demand_Hour_")]
#     if len(prev_day_cols) != y_pred_change_np.shape[1]: raise ValueError(f"Mismatch between number of previous day load columns ({len(prev_day_cols)}) and prediction columns ({y_pred_change_np.shape[1]})")
#     num_hours = y_pred_change_np.shape[1]; prev_day_indices = [X_original.columns.get_loc(col) for col in prev_day_cols]
#     prev_day_load = X_original_np[:, prev_day_indices]
#     y_pred_base_np = prev_day_load.astype(float) + y_pred_change_np.astype(float)
#     print("Converted change_in_load prediction back to base_load prediction.")
#     return y_pred_base_np


def train_ann_model(
    X_train_scaled,
    y_train_scaled,
    X_val_scaled,
    y_val_scaled,
    X_train_original,
    X_val_original,
    y_train_original,
    y_val_original,
    y_scaler,
    change_in_load=False,
    verbose=0,
):
    """Trains ANN model and returns predictions on train and val sets."""
    print(f"Training ANN model (change_in_load={change_in_load})...")

    X_mean = K.constant(X_train_scaled.mean(axis=0))
    X_std = K.constant(X_train_scaled.std(axis=0))
    X_min = K.constant(X_train_scaled.min(axis=0))
    X_max = K.constant(X_train_scaled.max(axis=0))

    y_mean = K.constant(y_train_scaled.mean(axis=0))
    y_std = K.constant(y_train_scaled.std(axis=0))
    y_min = K.constant(y_train_scaled.min(axis=0))
    y_max = K.constant(y_train_scaled.max(axis=0))

    ann_model = Sequential()
    ann_model.add(Input(shape=(X_train_scaled.shape[1],)))
    ann_model.add(NormalizeLayer(X_mean, X_std, normalize=True, name="input_normalization")),
    # ann_model.add(Lambda(lambda x: (x - X_mean) / (X_std + 1e-8), name="input_normalization"))
    Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.1))
    # ann_model.add(Dense(50, activation="sigmoid"))
    ann_model.add(Dropout(0.5))
    ann_model.add(Dense(y_train_scaled.shape[1]))
    ann_model.add(NormalizeLayer(y_mean, y_std, normalize=False, name="output_denormalization")),
    # ann_model.add(Lambda(lambda x: (x * y_std) + y_mean, name="output_denormalization"))

    ann_model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error", metrics=["mae"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=15, min_lr=1e-6, verbose=1),
    ]

    print("Fitting ANN on scaled data...")
    history = ann_model.fit(
        X_train_scaled, y_train_scaled, validation_data=(X_val_scaled, y_val_scaled), epochs=50, batch_size=32, callbacks=callbacks, verbose=verbose
    )
    print("Model fitting complete.")
    print("Predicting on scaled training set...")
    y_pred_train_scaled = ann_model.predict(X_train_scaled)
    print("Inverse transforming training predictions...")
    y_pred_train_original = y_scaler.inverse_transform(y_pred_train_scaled)
    if change_in_load:
        y_pred_train_original = convert_change_in_load_to_base_load(X_train_original, y_pred_train_original)
    print("Predicting on scaled validation set...")
    y_pred_val_scaled = ann_model.predict(X_val_scaled)
    print("Inverse transforming validation predictions...")
    y_pred_val_original = y_scaler.inverse_transform(y_pred_val_scaled)
    if change_in_load:
        y_pred_val_original = convert_change_in_load_to_base_load(X_val_original, y_pred_val_original)
    print("Calculating validation metrics on original scale...")
    y_val_np = y_val_original.values if isinstance(y_val_original, pd.DataFrame) else y_val_original
    y_pred_val_np = y_pred_val_original  # Already converted if needed
    try:
        mae = mean_absolute_error(y_val_np, y_pred_val_np)
        rmse = np.sqrt(mean_squared_error(y_val_np, y_pred_val_np))
        denominator = np.abs(y_val_np) + np.abs(y_pred_val_np)
        safe_denominator = np.where(denominator == 0, 1, denominator)
        smape_values = 200 * np.abs(y_pred_val_np - y_val_np) / safe_denominator
        smape = np.mean(smape_values)
        validation_metrics = {"mae": mae, "rmse": rmse, "smape": smape}
    except ValueError as metric_err:
        print(f"ERROR calculating metrics: {metric_err}")
        traceback.print_exc()
        validation_metrics = {"mae": np.nan, "rmse": np.nan, "smape": np.nan}
    print(f"ANN Model training complete. Validation Metrics (Original Scale): {validation_metrics}")
    return ann_model, validation_metrics, y_pred_train_original, y_pred_val_original


def train_lstm_model(
    X_train_scaled,
    y_train_scaled,
    X_val_scaled,
    y_val_scaled,
    X_train_original,
    X_val_original,
    y_train_original,
    y_val_original,
    y_scaler,
    change_in_load=False,
    verbose=0,
):
    """Trains LSTM model and returns predictions on train and val sets."""
    print(f"Training LSTM model (change_in_load={change_in_load})...")
    X_train_lstm = X_train_scaled.values.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_lstm = X_val_scaled.values.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
    print(f"Reshaped X for LSTM: Train shape {X_train_lstm.shape}, Val shape {X_val_lstm.shape}")
    print(f"NaN check before LSTM fit:")
    print(f"  X_train_lstm NaNs: {np.isnan(X_train_lstm).any()}")
    print(f"  y_train_scaled NaNs: {np.isnan(y_train_scaled.values).any()}")
    print(f"  X_val_lstm NaNs: {np.isnan(X_val_lstm).any()}")
    print(f"  y_val_scaled NaNs: {np.isnan(y_val_scaled.values).any()}")
    if np.isnan(X_train_lstm).any() or np.isnan(y_train_scaled.values).any():
        print("ERROR: NaNs detected in TRAINING data before LSTM fit!")
        return None, {"mae": np.nan, "rmse": np.nan, "smape": np.nan}, None, None
    if np.isnan(X_val_lstm).any() or np.isnan(y_val_scaled.values).any():
        print("ERROR: NaNs detected in VALIDATION data before LSTM fit!")

    # lstm_model = Sequential()
    # lstm_model.add(Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    # lstm_model.add(LSTM(50, return_sequences=False))
    # lstm_model.add(Dropout(0.3))
    # lstm_model.add(Dense(25, activation="relu"))
    # lstm_model.add(Dense(y_train_scaled.shape[1], activation="sigmoid"))

    print("Shape: ", X_train_lstm.shape)

    X_mean = K.constant(X_train_lstm.mean(axis=0))
    X_std = K.constant(X_train_lstm.std(axis=0))
    X_min = K.constant(X_train_lstm.min(axis=0))
    X_max = K.constant(X_train_lstm.max(axis=0))

    y_mean = K.constant(y_train_scaled.mean(axis=0))
    y_std = K.constant(y_train_scaled.std(axis=0))
    y_min = K.constant(y_train_scaled.min(axis=0))
    y_max = K.constant(y_train_scaled.max(axis=0))

    lstm_model = Sequential()

    lstm_model.add(Input(shape=(X_train_lstm.shape[1:])))
    # lstm_model.add(Lambda(lambda x: (x - X_mean) / (X_std + 1e-8), name="input_normalization"))
    lstm_model.add(NormalizeLayer(X_mean, X_std, normalize=True, name="input_normalization")),
    # Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.1))
    lstm_model.add(LSTM(128, return_sequences=False))
    # lstm_model.add(Dense(50, activation="sigmoid"))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(y_train_scaled.shape[1]))
    lstm_model.add(NormalizeLayer(y_mean, y_std, normalize=False, name="output_denormalization")),
    # lstm_model.add(Lambda(lambda x: (x * y_std) + y_mean, name="output_denormalization"))

    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error", metrics=["mae"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=15, min_lr=1e-6, verbose=1),
    ]

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, clipvalue=1.0)
    # lstm_model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    # lstm_model.summary()
    callbacks = [EarlyStopping(monitor="val_loss", patience=8, verbose=1, restore_best_weights=True)]
    print("Fitting LSTM on scaled data...")
    history = lstm_model.fit(
        X_train_lstm, y_train_scaled, validation_data=(X_val_lstm, y_val_scaled), epochs=50, batch_size=32, callbacks=callbacks, verbose=verbose
    )
    print("LSTM Model fitting complete.")
    if np.isnan(history.history["loss"]).any() or np.isnan(history.history["val_loss"]).any():
        print("WARNING: NaN detected in loss during training history.")
    print("Predicting on scaled training set with LSTM...")
    y_pred_train_scaled = lstm_model.predict(X_train_lstm)
    print("Inverse transforming training predictions...")
    y_pred_train_original = y_scaler.inverse_transform(y_pred_train_scaled)
    if change_in_load:
        y_pred_train_original = convert_change_in_load_to_base_load(X_train_original, y_pred_train_original)
    print("Predicting on scaled validation set with LSTM...")
    y_pred_val_scaled = lstm_model.predict(X_val_lstm)
    print("Inverse transforming validation predictions...")
    y_pred_val_original = y_scaler.inverse_transform(y_pred_val_scaled)
    if change_in_load:
        y_pred_val_original = convert_change_in_load_to_base_load(X_val_original, y_pred_val_original)
    print("Calculating LSTM validation metrics on original scale...")
    y_val_np = y_val_original.values if isinstance(y_val_original, pd.DataFrame) else y_val_original
    y_pred_val_np = y_pred_val_original  # Already converted if needed
    try:
        if np.isnan(y_val_np).any():
            raise ValueError("NaN found in y_val_np for metrics")
        if np.isnan(y_pred_val_np).any():
            raise ValueError("NaN found in y_pred_val_np for metrics")
        mae = mean_absolute_error(y_val_np, y_pred_val_np)
        rmse = np.sqrt(mean_squared_error(y_val_np, y_pred_val_np))
        denominator = np.abs(y_val_np) + np.abs(y_pred_val_np)
        safe_denominator = np.where(denominator == 0, 1, denominator)
        smape_values = 200 * np.abs(y_pred_val_np - y_val_np) / safe_denominator
        smape = np.mean(smape_values)
        validation_metrics = {"mae": mae, "rmse": rmse, "smape": smape}
    except ValueError as metric_err:
        print(f"ERROR calculating metrics: {metric_err}")
        traceback.print_exc()
        validation_metrics = {"mae": np.nan, "rmse": np.nan, "smape": np.nan}
    print(f"LSTM Model training complete. Validation Metrics (Original Scale): {validation_metrics}")
    return lstm_model, validation_metrics, y_pred_train_original, y_pred_val_original


def train_padasip_rls_combiner(predictions1, predictions2, actuals, mu=0.98, eps=0.01):
    """Trains a list of padasip FilterRLS objects, one for each output hour."""
    n_samples, n_outputs = actuals.shape
    n_inputs = 2
    print(f"Initializing {n_outputs} RLS filters (mu={mu}, eps={eps})...")
    rls_filters = [FilterRLS(n=n_inputs, mu=mu) for _ in range(n_outputs)]

    # return rls_filters
    print("Adapting RLS filters sample by sample...")
    for t in range(n_samples):  # Iterate through all samples
        for k in range(n_outputs):
            x_k = np.array([predictions1[t, k], predictions2[t, k]])
            d_k = actuals[t, k]
            try:
                if np.isnan(x_k).any() or np.isinf(x_k).any() or np.isnan(d_k) or np.isinf(d_k):
                    print(f"Warning: Skipping RLS adapt at sample {t}, hour {k} due to NaN/Inf input/target.")
                    continue
                rls_filters[k].adapt(d_k, x_k)
            except Exception as adapt_err:
                print(f"ERROR during RLS adapt at sample {t}, hour {k}: {adapt_err}")
                pass
    print("RLS filter adaptation complete.")
    return rls_filters


# def predict_with_padasip_rls(rls_filters, predictions1, predictions2):
#     """Combines predictions using a list of fitted padasip RLS filters."""
#     n_samples, n_outputs = predictions1.shape
#     if len(rls_filters) != n_outputs: raise ValueError("Number of RLS filters does not match number of prediction outputs.")
#     combined_predictions = np.zeros_like(predictions1)
#     for t in range(n_samples):
#         for k in range(n_outputs):
#             x_k = np.array([predictions1[t, k], predictions2[t, k]])
#             combined_predictions[t, k] = rls_filters[k].predict(x_k)
#     return combined_predictions


# --- Helper function for RLS stages (Modified) ---


def predict_with_loaded_artifact(loaded_artifact, X_scaled_input, X_original_input, predicts_change):
    """
    Generates predictions using a loaded model artifact (model, scalers, columns).

    Args:
        loaded_artifact (dict): Dictionary containing 'model', 'x_scaler', 'y_scaler', 'feature_columns'.
        X_scaled_input (pd.DataFrame): Scaled input features for prediction.
        X_original_input (pd.DataFrame): Original (unscaled) input features (needed for change conversion).
        predicts_change (bool): Flag indicating if the model predicts change_in_load.

    Returns:
        np.array: Predictions in the original scale. Returns None on error.
    """
    print(f"Generating predictions using loaded artifact...")
    try:
        model = loaded_artifact.get("model")
        # x_scaler = loaded_artifact.get('x_scaler') # Not needed for prediction itself, only for prep
        y_scaler = loaded_artifact.get("y_scaler")
        feature_columns = loaded_artifact.get("feature_columns")
        target_columns = loaded_artifact.get("target_columns")  # Needed for shape check

        if model is None or y_scaler is None or feature_columns is None or target_columns is None:
            raise ValueError("Loaded artifact is missing required components (model, y_scaler, feature_columns, target_columns).")

        # --- Ensure Input Columns Match ---
        if list(X_scaled_input.columns) != feature_columns:
            print("Warning: Input feature columns mismatch artifact. Attempting reorder.")
            try:
                X_scaled_input = X_scaled_input[feature_columns]
            except KeyError as ke:
                missing = set(feature_columns) - set(X_scaled_input.columns)
                extra = set(X_scaled_input.columns) - set(feature_columns)
                raise ValueError(f"Feature mismatch during prediction. Missing: {missing}. Extra: {extra}") from ke

        # --- Predict Scaled Values ---
        print(f"Predicting scaled values with model type: {type(model)}...")
        is_lstm = KERAS_AVAILABLE and isinstance(model, tf.keras.Model) and any(isinstance(layer, tf.keras.layers.LSTM) for layer in model.layers)
        if is_lstm:
            print("Reshaping input for LSTM prediction...")
            # Ensure input has samples dimension even if it's just one day
            num_samples = X_scaled_input.shape[0]
            X_input_final = X_scaled_input.values.reshape((num_samples, 1, X_scaled_input.shape[1]))
        else:
            X_input_final = X_scaled_input.values

        y_pred_scaled = model.predict(X_input_final)
        print(f"Scaled prediction shape: {y_pred_scaled.shape}")

        # --- Inverse Transform ---
        print("Inverse transforming predictions...")
        # Infer expected output shape from target_columns
        expected_outputs = len(target_columns)
        if y_pred_scaled.shape[1] != expected_outputs:
            # Attempt reshape if prediction is flat but expecting 2D
            if len(y_pred_scaled.shape) == 1 and y_pred_scaled.shape[0] == expected_outputs * X_scaled_input.shape[0]:
                print(
                    f"Warning: Reshaping flat prediction ({y_pred_scaled.shape}) to ({X_scaled_input.shape[0]}, {expected_outputs}) for inverse transform."
                )
                y_pred_scaled = y_pred_scaled.reshape((X_scaled_input.shape[0], expected_outputs))
            else:
                raise ValueError(
                    f"Prediction shape mismatch for inverse transform: predicted {y_pred_scaled.shape[1]} features, expected {expected_outputs} based on target columns."
                )

        if y_pred_scaled.shape[1] != y_scaler.n_features_in_:
            raise ValueError(f"Prediction shape ({y_pred_scaled.shape[1]}) mismatch y_scaler features ({y_scaler.n_features_in_}).")

        y_pred_original = y_scaler.inverse_transform(y_pred_scaled)

        # --- Post-process (Convert Change to Base if needed) ---
        final_prediction_original = y_pred_original
        if predicts_change:
            print("Converting change_in_load prediction...")
            # Ensure X_original_input is aligned with X_scaled_input index
            X_original_aligned = X_original_input.loc[X_scaled_input.index]
            if list(X_original_aligned.columns) != feature_columns:
                # Reorder original columns too if needed (though less likely to mismatch if prep was consistent)
                print("Reordering original X columns for change conversion...")
                X_original_aligned = X_original_aligned[feature_columns]

            final_prediction_original = convert_change_in_load_to_base_load(X_original_aligned, y_pred_original)

        print("Prediction generation complete.")
        return final_prediction_original

    except Exception as e:
        print(f"ERROR during predict_with_loaded_artifact: {e}")
        traceback.print_exc()
        return None  # Return None on error


# def run_rls_combination_stage(train_df_raw, val_df_raw, scenario, model_type):
#     """
#     Runs the RLS combination for either ANN or LSTM.
#     Trains base models, adapts RLS on training preds, predicts RLS on validation preds.
#     Returns:
#         y_pred_rls_combined_train (np.array): RLS predictions on training set.
#         y_pred_rls_combined_val (np.array): RLS predictions on validation set.
#         y_val_base_orig (pd.DataFrame): Original validation actuals (base load).
#         rls_filter_list (list): List of RLS filters fitted on training data.
#     """
#     print(f"\n--- Running Internal {model_type} RLS Combination Stage ---")
#     base_model_func = train_ann_model if model_type == "ANN" else train_lstm_model
#     change_model_func = train_ann_model if model_type == "ANN" else train_lstm_model
#     rls_filter_list = None

#     try:
#         # --- Prepare Data ---
#         # Train Data
#         X_train_base_s, y_train_base_s, x_scaler_base, y_scaler_base = feature_engineer_and_scale(train_df_raw, scenario, change_in_load=False)
#         X_train_change_s, y_train_change_s, x_scaler_change, y_scaler_change = feature_engineer_and_scale(train_df_raw, scenario, change_in_load=True)
#         X_train_base_orig, y_train_base_orig, _, _ = feature_engineer_and_scale(
#             train_df_raw, scenario, x_scaler=x_scaler_base, y_scaler=y_scaler_base, change_in_load=False, apply_scaling=False
#         )
#         X_train_change_orig, _, _, _ = feature_engineer_and_scale(
#             train_df_raw, scenario, x_scaler=x_scaler_change, y_scaler=y_scaler_change, change_in_load=True, apply_scaling=False
#         )
#         # Align original train data
#         common_train_index = y_train_base_s.index.intersection(y_train_change_s.index)
#         X_train_base_orig = X_train_base_orig.loc[common_train_index]
#         y_train_base_orig = y_train_base_orig.loc[common_train_index]
#         X_train_change_orig = X_train_change_orig.loc[common_train_index]
#         X_train_base_s = X_train_base_s.loc[common_train_index]
#         y_train_base_s = y_train_base_s.loc[common_train_index]
#         X_train_change_s = X_train_change_s.loc[common_train_index]
#         y_train_change_s = y_train_change_s.loc[common_train_index]

#         # Validation Data
#         X_val_base_s, y_val_base_s, _, _ = feature_engineer_and_scale(
#             val_df_raw, scenario, x_scaler=x_scaler_base, y_scaler=y_scaler_base, change_in_load=False
#         )
#         X_val_change_s, y_val_change_s, _, _ = feature_engineer_and_scale(
#             val_df_raw, scenario, x_scaler=x_scaler_change, y_scaler=y_scaler_change, change_in_load=True
#         )
#         X_val_base_orig, y_val_base_orig, _, _ = feature_engineer_and_scale(
#             val_df_raw, scenario, x_scaler=x_scaler_base, y_scaler=y_scaler_base, change_in_load=False, apply_scaling=False
#         )
#         X_val_change_orig, _, _, _ = feature_engineer_and_scale(
#             val_df_raw, scenario, x_scaler=x_scaler_change, y_scaler=y_scaler_change, change_in_load=True, apply_scaling=False
#         )
#         # Align original val data
#         common_val_index = y_val_base_s.index.intersection(y_val_change_s.index)
#         X_val_base_orig = X_val_base_orig.loc[common_val_index]
#         y_val_base_orig = y_val_base_orig.loc[common_val_index]
#         X_val_change_orig = X_val_change_orig.loc[common_val_index]
#         X_val_base_s = X_val_base_s.loc[common_val_index]
#         y_val_base_s = y_val_base_s.loc[common_val_index]
#         X_val_change_s = X_val_change_s.loc[common_val_index]
#         y_val_change_s = y_val_change_s.loc[common_val_index]

#         # --- Train Base Models & Get Train+Val Predictions ---
#         # Base model
#         base_model, _, y_pred_base_train_orig, y_pred_base_val_orig = base_model_func(
#             X_train_base_s,
#             y_train_base_s,
#             X_val_base_s,
#             y_val_base_s,
#             X_train_base_orig,
#             X_val_base_orig,
#             y_train_base_orig,
#             y_val_base_orig,  # Pass train originals too
#             y_scaler_base,
#             change_in_load=False,
#         )
#         if base_model is None:
#             raise RuntimeError(f"{model_type}_Baseload training failed.")

#         # Change model
#         change_model, _, y_pred_change_converted_train_orig, y_pred_change_converted_val_orig = change_model_func(
#             X_train_change_s,
#             y_train_change_s,
#             X_val_change_s,
#             y_val_change_s,
#             X_train_change_orig,
#             X_val_change_orig,
#             y_train_base_orig,
#             y_val_base_orig,  # Pass train originals, use BASE actuals for metrics
#             y_scaler_change,
#             change_in_load=True,
#         )
#         if change_model is None:
#             raise RuntimeError(f"{model_type}_Change_in_Load training failed.")

#         # --- Convert Change Predictions (Train & Val) ---
#         # y_pred_change_converted_train_orig = convert_change_in_load_to_base_load(X_train_change_orig, y_pred_change_train_orig)
#         # y_pred_change_converted_val_orig = convert_change_in_load_to_base_load(X_val_change_orig, y_pred_change_val_orig)

#         # --- Align Training Data for RLS Adaptation ---
#         y_pred_base_train_aligned_np = pd.DataFrame(y_pred_base_train_orig, index=common_train_index).reindex(common_train_index).values
#         y_pred_change_converted_train_aligned_np = (
#             pd.DataFrame(y_pred_change_converted_train_orig, index=common_train_index).reindex(common_train_index).values
#         )
#         y_train_base_orig_np = y_train_base_orig.reindex(common_train_index).values

#         # --- Train RLS Filters on Training Data ---
#         print(f"\n--- Training {model_type} RLS Combiner on Training Data ---")
#         rls_filter_list = train_padasip_rls_combiner(
#             y_pred_base_train_aligned_np,
#             y_pred_change_converted_train_aligned_np,
#             y_train_base_orig_np,
#             mu=0.5,
#             eps=0.1,  # Use defaults for intermediate stage
#         )

#         # --- Predict RLS on Validation Data ---
#         print(f"\n--- Predicting {model_type} RLS Combiner on Validation Data ---")
#         y_pred_base_val_aligned_np = pd.DataFrame(y_pred_base_val_orig, index=common_val_index).reindex(common_val_index).values
#         y_pred_change_converted_val_aligned_np = (
#             pd.DataFrame(y_pred_change_converted_val_orig, index=common_val_index).reindex(common_val_index).values
#         )

#         y_pred_rls_combined_val = predict_with_padasip_rls(
#             rls_filter_list, y_val_base_orig.values, y_pred_base_val_aligned_np, y_pred_change_converted_val_aligned_np
#         )

#         # --- Predict RLS on Training Data (Needed for Final Combiner) ---
#         print(f"\n--- Predicting {model_type} RLS Combiner on Training Data ---")
#         y_pred_rls_combined_train = predict_with_padasip_rls(
#             rls_filter_list, y_train_base_orig.values, y_pred_base_train_aligned_np, y_pred_change_converted_train_aligned_np
#         )

#         print(f"--- Internal {model_type} RLS Combination Stage Complete ---")
#         # Return TRAIN RLS preds, VAL RLS preds, VAL actuals df, fitted filters
#         return y_pred_rls_combined_train, y_pred_rls_combined_val, y_val_base_orig, rls_filter_list

#     except Exception as e:
#         print(f"Error during internal {model_type} RLS combination stage: {e}")
#         traceback.print_exc()
#         return None, None, None, None


# # --- Helper function for RLS stages (MODIFIED TO LOAD MODELS) ---
# def run_rls_combination_stage(train_df_raw, val_df_raw, scenario, model_type, feeder_id, version):
#     """
#     Runs the RLS combination stage by LOADING pre-trained base/change models.
#     Adapts RLS on training preds, predicts RLS on validation preds.

#     Args:
#         train_df_raw, val_df_raw: Raw dataframes.
#         scenario (str): '24hr', 'Day', or 'Night'.
#         model_type (str): 'ANN' or 'LSTM'.
#         feeder_id (int): The feeder ID.
#         version (str): The EXACT version tag to load for base/change models.

#     Returns:
#         y_pred_rls_combined_train (np.array): RLS predictions on training set.
#         y_pred_rls_combined_val (np.array): RLS predictions on validation set.
#         y_val_base_orig_df (pd.DataFrame): Original validation actuals (base load).
#         rls_filter_list (list): List of RLS filters fitted on training data.
#     """
#     print(f"\n--- Running Internal {model_type} RLS Combination Stage (Loading Models Version: {version}) ---")
#     rls_filter_list = None
#     base_model_arch = f"{model_type}_Baseload"
#     change_model_arch = f"{model_type}_Change_in_Load"

#     try:
#         # --- Find and Load Base Model Artifact ---
#         print(f"Loading Base Model: {base_model_arch}...")
#         base_metadata = select_model_for_forecast(feeder_id, base_model_arch, scenario, version)
#         if not base_metadata:
#             raise FileNotFoundError(
#                 f"Required base model not found: Feeder={feeder_id}, Arch={base_model_arch}, Scenario={scenario}, Version={version}"
#             )
#         loaded_base_artifact = load_artifact_from_storage(base_metadata["model_artifact_path"])
#         if loaded_base_artifact is None or "model" not in loaded_base_artifact:
#             raise ValueError(f"Failed to load valid artifact for {base_model_arch}")

#         # --- Find and Load Change Model Artifact ---
#         print(f"Loading Change Model: {change_model_arch}...")
#         change_metadata = select_model_for_forecast(feeder_id, change_model_arch, scenario, version)
#         if not change_metadata:
#             raise FileNotFoundError(
#                 f"Required change model not found: Feeder={feeder_id}, Arch={change_model_arch}, Scenario={scenario}, Version={version}"
#             )
#         loaded_change_artifact = load_artifact_from_storage(change_metadata["model_artifact_path"])
#         if loaded_change_artifact is None or "model" not in loaded_change_artifact:
#             raise ValueError(f"Failed to load valid artifact for {change_model_arch}")

#         # --- Prepare Data (Need scaled X and original X/y for predictions) ---
#         # Use scalers from the LOADED base model artifact for consistency
#         base_x_scaler = loaded_base_artifact["x_scaler"]
#         base_y_scaler = loaded_base_artifact["y_scaler"]  # Needed for feature_engineer_and_scale call structure
#         if base_x_scaler is None or base_y_scaler is None:
#             raise ValueError("Base model artifact missing scalers.")

#         # Prepare Training Data Inputs
#         X_train_scaled, _, _, _ = feature_engineer_and_scale(
#             train_df_raw, scenario, x_scaler=base_x_scaler, y_scaler=base_y_scaler, change_in_load=False, apply_scaling=True
#         )
#         X_train_original, y_train_base_orig_df, _, _ = feature_engineer_and_scale(
#             train_df_raw, scenario, x_scaler=base_x_scaler, y_scaler=base_y_scaler, change_in_load=False, apply_scaling=False
#         )

#         # Prepare Validation Data Inputs
#         X_val_scaled, _, _, _ = feature_engineer_and_scale(
#             val_df_raw, scenario, x_scaler=base_x_scaler, y_scaler=base_y_scaler, change_in_load=False, apply_scaling=True
#         )
#         X_val_original, y_val_base_orig_df, _, _ = feature_engineer_and_scale(
#             val_df_raw, scenario, x_scaler=base_x_scaler, y_scaler=base_y_scaler, change_in_load=False, apply_scaling=False
#         )

#         # Align indices after potential drops during feature engineering
#         common_train_index = X_train_scaled.index.intersection(X_train_original.index).intersection(y_train_base_orig_df.index)
#         X_train_scaled = X_train_scaled.loc[common_train_index]
#         X_train_original = X_train_original.loc[common_train_index]
#         y_train_base_orig_df = y_train_base_orig_df.loc[common_train_index]

#         common_val_index = X_val_scaled.index.intersection(X_val_original.index).intersection(y_val_base_orig_df.index)
#         X_val_scaled = X_val_scaled.loc[common_val_index]
#         X_val_original = X_val_original.loc[common_val_index]
#         y_val_base_orig_df = y_val_base_orig_df.loc[common_val_index]

#         if X_train_scaled.empty or X_val_scaled.empty:
#             raise ValueError("Data preparation resulted in empty DataFrames.")

#         # --- Generate Predictions using LOADED Models ---
#         print("\n--- Generating predictions for RLS training/validation ---")
#         # Predict Base Model (Train & Val)
#         y_pred_base_train_orig = predict_with_loaded_artifact(loaded_base_artifact, X_train_scaled, X_train_original, predicts_change=False)
#         y_pred_base_val_orig = predict_with_loaded_artifact(loaded_base_artifact, X_val_scaled, X_val_original, predicts_change=False)

#         # Predict Change Model (Train & Val) - it will convert to base load internally
#         y_pred_change_converted_train_orig = predict_with_loaded_artifact(
#             loaded_change_artifact, X_train_scaled, X_train_original, predicts_change=True
#         )
#         y_pred_change_converted_val_orig = predict_with_loaded_artifact(loaded_change_artifact, X_val_scaled, X_val_original, predicts_change=True)

#         if (
#             y_pred_base_train_orig is None
#             or y_pred_base_val_orig is None
#             or y_pred_change_converted_train_orig is None
#             or y_pred_change_converted_val_orig is None
#         ):
#             raise RuntimeError(f"Failed to generate predictions from loaded base/change models.")

#         # --- Align Training Data for RLS Adaptation ---
#         # Predictions are already numpy arrays in original scale
#         y_train_base_orig_np = y_train_base_orig_df.values

#         # --- Train RLS Filters on Training Data ---
#         print(f"\n--- Training {model_type} RLS Combiner on Training Data ---")
#         rls_filter_list = train_padasip_rls_combiner(
#             y_pred_base_train_orig,  # Use train predictions
#             y_pred_change_converted_train_orig,  # Use train predictions
#             y_train_base_orig_np,  # Use train actuals
#             mu=0.5,
#             eps=0.1,
#         )

#         # --- Predict RLS on Validation Data ---
#         print(f"\n--- Predicting {model_type} RLS Combiner on Validation Data ---")
#         y_val_base_orig_np = y_val_base_orig_df.values
#         y_pred_rls_combined_val = predict_with_padasip_rls(
#             rls_filter_list,
#             y_val_base_orig_np,  # Pass validation actuals for adaptation
#             y_pred_base_val_orig,  # Use validation predictions
#             y_pred_change_converted_val_orig,  # Use validation predictions
#         )

#         # --- Predict RLS on Training Data (Needed for Final Combiner) ---
#         print(f"\n--- Predicting {model_type} RLS Combiner on Training Data ---")
#         y_pred_rls_combined_train = predict_with_padasip_rls(
#             rls_filter_list,
#             y_train_base_orig_np,  # Pass training actuals for adaptation
#             y_pred_base_train_orig,  # Use training predictions
#             y_pred_change_converted_train_orig,  # Use training predictions
#         )

#         print(f"--- Internal {model_type} RLS Combination Stage Complete ---")
#         # Return TRAIN RLS preds, VAL RLS preds, VAL actuals df, fitted filters
#         return y_pred_rls_combined_train, y_pred_rls_combined_val, y_val_base_orig_df, rls_filter_list

#     except FileNotFoundError as e:
#         print(f"ERROR: Required input model artifact not found: {e}")
#         print("Ensure base/change models with the exact same version tag were trained and saved successfully first.")
#         traceback.print_exc()
#         return None, None, None, None  # Indicate failure
#     except Exception as e:
#         print(f"Error during internal {model_type} RLS combination stage (loading models): {e}")
#         traceback.print_exc()
#         return None, None, None, None  # Indicate failure

# Trainer_Utils.py


# --- Helper function for RLS stages (MODIFIED TO SCALE BEFORE RLS) ---
def run_rls_combination_stage(train_df_raw, val_df_raw, scenario, model_type, feeder_id, version):
    """
    Runs the RLS combination stage by LOADING pre-trained base/change models.
    SCALES predictions/actuals before adapting RLS filters.
    Predicts RLS on validation preds (using internal scaling).

    Args:
        train_df_raw, val_df_raw: Raw dataframes.
        scenario (str): '24hr', 'Day', or 'Night'.
        model_type (str): 'ANN' or 'LSTM'.
        feeder_id (int): The feeder ID.
        version (str): The EXACT version tag to load for base/change models.

    Returns:
        y_pred_rls_combined_train_orig (np.array): RLS predictions on training set (original scale).
        y_pred_rls_combined_val_orig (np.array): RLS predictions on validation set (original scale).
        y_val_base_orig_df (pd.DataFrame): Original validation actuals (base load).
        rls_filter_list (list): List of RLS filters fitted on SCALED training data.
    """
    print(f"\n--- Running Internal {model_type} RLS Combination Stage (Loading Models V:{version}, Scaling for RLS) ---")
    rls_filter_list = None
    base_model_arch = f"{model_type}_Baseload"
    change_model_arch = f"{model_type}_Change_in_Load"
    y_scaler_for_rls = None  # Store the scaler to use

    try:
        # --- Load Base Model Artifact (also gets the y_scaler) ---
        print(f"Loading Base Model: {base_model_arch}...")
        base_metadata = select_model_for_forecast(feeder_id, base_model_arch, scenario, version)
        if not base_metadata:
            raise FileNotFoundError(
                f"Required base model not found: Feeder={feeder_id}, Arch={base_model_arch}, Scenario={scenario}, Version={version}"
            )
        loaded_base_artifact = load_artifact_from_storage(base_metadata["model_artifact_path"])
        if loaded_base_artifact is None or "model" not in loaded_base_artifact:
            raise ValueError(f"Failed to load valid artifact for {base_model_arch}")
        y_scaler_for_rls = loaded_base_artifact.get("y_scaler")  # Get the scaler
        base_x_scaler = loaded_base_artifact.get("x_scaler")
        if y_scaler_for_rls is None or base_x_scaler is None:
            raise ValueError("Base model artifact missing required scalers.")

        # --- Load Change Model Artifact ---
        print(f"Loading Change Model: {change_model_arch}...")
        change_metadata = select_model_for_forecast(feeder_id, change_model_arch, scenario, version)
        if not change_metadata:
            raise FileNotFoundError(
                f"Required change model not found: Feeder={feeder_id}, Arch={change_model_arch}, Scenario={scenario}, Version={version}"
            )
        loaded_change_artifact = load_artifact_from_storage(change_metadata["model_artifact_path"])
        if loaded_change_artifact is None or "model" not in loaded_change_artifact:
            raise ValueError(f"Failed to load valid artifact for {change_model_arch}")

        # --- Prepare Data (Scaled X for input, Original X/y for context/actuals) ---
        # Prepare Training Data Inputs
        X_train_scaled, _, _, _ = feature_engineer_and_scale(
            train_df_raw, scenario, x_scaler=base_x_scaler, y_scaler=y_scaler_for_rls, change_in_load=False, apply_scaling=True
        )
        X_train_original, y_train_base_orig_df, _, _ = feature_engineer_and_scale(
            train_df_raw, scenario, x_scaler=base_x_scaler, y_scaler=y_scaler_for_rls, change_in_load=False, apply_scaling=False
        )
        # Prepare Validation Data Inputs
        X_val_scaled, _, _, _ = feature_engineer_and_scale(
            val_df_raw, scenario, x_scaler=base_x_scaler, y_scaler=y_scaler_for_rls, change_in_load=False, apply_scaling=True
        )
        X_val_original, y_val_base_orig_df, _, _ = feature_engineer_and_scale(
            val_df_raw, scenario, x_scaler=base_x_scaler, y_scaler=y_scaler_for_rls, change_in_load=False, apply_scaling=False
        )
        # Align indices
        common_train_index = X_train_scaled.index.intersection(X_train_original.index).intersection(y_train_base_orig_df.index)
        X_train_scaled = X_train_scaled.loc[common_train_index]
        X_train_original = X_train_original.loc[common_train_index]
        y_train_base_orig_df = y_train_base_orig_df.loc[common_train_index]
        common_val_index = X_val_scaled.index.intersection(X_val_original.index).intersection(y_val_base_orig_df.index)
        X_val_scaled = X_val_scaled.loc[common_val_index]
        X_val_original = X_val_original.loc[common_val_index]
        y_val_base_orig_df = y_val_base_orig_df.loc[common_val_index]
        if X_train_scaled.empty or X_val_scaled.empty:
            raise ValueError("Data preparation resulted in empty DataFrames.")

        # --- Generate Predictions using LOADED Models (Original Scale) ---
        print("\n--- Generating predictions (original scale) for RLS training/validation ---")
        y_pred_base_train_orig = predict_with_loaded_artifact(loaded_base_artifact, X_train_scaled, X_train_original, predicts_change=False)
        y_pred_base_val_orig = predict_with_loaded_artifact(loaded_base_artifact, X_val_scaled, X_val_original, predicts_change=False)
        y_pred_change_converted_train_orig = predict_with_loaded_artifact(
            loaded_change_artifact, X_train_scaled, X_train_original, predicts_change=True
        )
        y_pred_change_converted_val_orig = predict_with_loaded_artifact(loaded_change_artifact, X_val_scaled, X_val_original, predicts_change=True)
        if y_pred_base_train_orig is None or y_pred_change_converted_train_orig is None:
            raise RuntimeError(f"Failed to generate training predictions from loaded models.")
        if y_pred_base_val_orig is None or y_pred_change_converted_val_orig is None:
            raise RuntimeError(f"Failed to generate validation predictions from loaded models.")

        # --- Scale Predictions and Actuals for RLS Training ---
        print("\n--- Scaling data for RLS Filter Training ---")
        y_train_base_orig_np = y_train_base_orig_df.values
        try:
            y_train_actuals_scaled = y_scaler_for_rls.transform(y_train_base_orig_np)
            y_pred_base_train_scaled = y_scaler_for_rls.transform(y_pred_base_train_orig)
            y_pred_change_train_scaled = y_scaler_for_rls.transform(y_pred_change_converted_train_orig)
        except Exception as scale_err:
            raise ValueError("Failed to scale training data/predictions for RLS.") from scale_err

        # --- Train RLS Filters on SCALED Training Data ---
        print(f"\n--- Training {model_type} RLS Combiner on SCALED Training Data ---")
        rls_filter_list = train_padasip_rls_combiner(
            y_pred_base_train_scaled,  # Use SCALED train predictions
            y_pred_change_train_scaled,  # Use SCALED train predictions
            y_train_actuals_scaled,  # Use SCALED train actuals
            mu=0.99,
            eps=0.1,
        )

        # --- Predict RLS on Validation Data (using modified predict function) ---
        print(f"\n--- Predicting {model_type} RLS Combiner on Validation Data (with internal scaling) ---")
        y_val_base_orig_np = y_val_base_orig_df.values
        # Pass ORIGINAL scale preds/actuals and the scaler to the modified predict function
        y_pred_rls_combined_val_orig = predict_with_padasip_rls(
            rls_filter_list,
            y_scaler_for_rls,  # Pass the scaler
            y_val_base_orig_np,  # Pass validation actuals (original)
            y_pred_base_val_orig,  # Pass validation predictions (original)
            y_pred_change_converted_val_orig,  # Pass validation predictions (original)
        )

        # --- Predict RLS on Training Data (Needed for Final Combiner) ---
        print(f"\n--- Predicting {model_type} RLS Combiner on Training Data (with internal scaling) ---")
        # Pass ORIGINAL scale preds/actuals and the scaler
        y_pred_rls_combined_train_orig = predict_with_padasip_rls(
            rls_filter_list,
            y_scaler_for_rls,  # Pass the scaler
            y_train_base_orig_np,  # Pass training actuals (original)
            y_pred_base_train_orig,  # Pass training predictions (original)
            y_pred_change_converted_train_orig,  # Pass training predictions (original)
        )

        print(f"--- Internal {model_type} RLS Combination Stage Complete ---")
        # Return ORIGINAL scale RLS preds, VAL actuals df, fitted filters
        return y_pred_rls_combined_train_orig, y_pred_rls_combined_val_orig, y_val_base_orig_df, rls_filter_list

    # ... (keep existing exception handling) ...
    except FileNotFoundError as e:
        print(f"ERROR: Required input model artifact not found: {e}")
        print("Ensure base/change models with the exact same version tag were trained and saved successfully first.")
        traceback.print_exc()
        return None, None, None, None  # Indicate failure
    except Exception as e:
        print(f"Error during internal {model_type} RLS combination stage (loading models): {e}")
        traceback.print_exc()
        return None, None, None, None  # Indicate failure


# --- Main Training Workflow (Modified Saving Logic) ---
def run_training(feeder_id, model_arch, scenario, version, train_start, train_end, val_start, val_end):
    """Orchestrates the training process for different architectures."""
    print(f"\n--- Starting Training Run ---")
    print(f"Feeder: {feeder_id}, Arch: {model_arch}, Scenario: {scenario}, Version: {version}")
    print(f"Train Period: {train_start} to {train_end}")
    print(f"Validation Period: {val_start} to {val_end}")

    # Determine if the architecture uses Keras
    is_keras_model = model_arch in ["ANN_Baseload", "ANN_Change_in_Load", "LSTM_Baseload", "LSTM_Change_in_Load"]
    if is_keras_model and not KERAS_AVAILABLE:
        print(f"ERROR: Keras is required for architecture '{model_arch}' but is not installed.")
        return

    is_single_model_run = model_arch in ["LightGBM_Baseline", "ANN_Baseload", "ANN_Change_in_Load", "LSTM_Baseload", "LSTM_Change_in_Load"]
    is_ann_rls_run = model_arch == "ANN_RLS_Combined"
    is_lstm_rls_run = model_arch == "LSTM_RLS_Combined"
    is_final_rls_run = model_arch == "Final_RLS_Combined"

    fetch_train_start = (pd.to_datetime(train_start) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S%z")
    train_df_raw = fetch_data(feeder_id, fetch_train_start, train_end)
    fetch_val_start = (pd.to_datetime(val_start) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S%z")
    val_df_raw = fetch_data(feeder_id, fetch_val_start, val_end)
    if train_df_raw.empty or val_df_raw.empty:
        print("Insufficient raw data fetched. Aborting training.")
        return

    model_object_trained = None  # Holds the primary trained object (model or filters)
    final_validation_metrics = None
    hyperparameters = {}
    feature_config = {}
    y_pred_val_plot = None
    y_val_original = None  # Will store the appropriate actuals df for plotting
    # Variables needed for saving scalers separately
    fitted_x_scaler = None
    fitted_y_scaler = None
    feature_columns_list = None
    target_columns_list = None

    try:
        # =============================================
        # === Handle Single Model Architectures =====
        # =============================================
        if is_single_model_run:
            change_in_load = model_arch in ["ANN_Change_in_Load", "LSTM_Change_in_Load"]
            apply_scaling = True  # Always scale for these models

            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("TESTING: ", model_arch)
            print("TESTING: ", is_single_model_run)
            print("TESTING: ", change_in_load)
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

            # print("Change in load: ", change_in_load, model_arch)

            # Perform feature engineering and scaling
            X_train_scaled, y_train_scaled, fitted_x_scaler, fitted_y_scaler = feature_engineer_and_scale(
                train_df_raw, scenario, x_scaler=None, y_scaler=None, change_in_load=change_in_load, apply_scaling=apply_scaling
            )
            X_train_original, y_train_original_base, _, _ = feature_engineer_and_scale(
                train_df_raw, scenario, x_scaler=fitted_x_scaler, y_scaler=fitted_y_scaler, change_in_load=False, apply_scaling=False
            )

            X_train_change, y_train_change_base, _, _ = feature_engineer_and_scale(
                train_df_raw, scenario, x_scaler=fitted_x_scaler, y_scaler=fitted_y_scaler, change_in_load=True, apply_scaling=False
            )

            y_train_original_base = y_train_original_base.loc[y_train_scaled.index]  # Align
            X_val_scaled, y_val_scaled, _, _ = feature_engineer_and_scale(
                val_df_raw, scenario, x_scaler=fitted_x_scaler, y_scaler=fitted_y_scaler, change_in_load=change_in_load, apply_scaling=apply_scaling
            )
            X_val_original, y_val_original_base, _, _ = feature_engineer_and_scale(
                val_df_raw, scenario, x_scaler=fitted_x_scaler, y_scaler=fitted_y_scaler, change_in_load=False, apply_scaling=False
            )

            X_val_change, y_val_change_base, _, _ = feature_engineer_and_scale(
                val_df_raw, scenario, x_scaler=fitted_x_scaler, y_scaler=fitted_y_scaler, change_in_load=True, apply_scaling=False
            )
            y_val_original = y_val_original_base.loc[y_val_scaled.index]  # Actuals for plotting/metrics are base load

            # print("Original train data shape:", X_train_change.shape, y_train_change_base.shape)
            # print("change train data values:", y_train_change_base.head(10))
            # y_train_change_base.to_csv("y_train_change_base.csv", index=True)  # Save for debugging
            # X_train_change.to_csv("X_train_change_base.csv", index=True)  # Save for debugging
            # X_train_original.to_csv("X_train_original_base.csv", index=True)  # Save for debugging
            # print("change val data shape:", X_val_change.shape, y_val_change_base.shape)
            # print("change val data values:", y_val_change_base.head(10))
            # y_val_change_base.to_csv("y_val_change_base.csv", index=True)  # Save for debugging
            # X_val_change.to_csv("X_val_change_base.csv", index=True)  # Save for debugging
            # X_val_original.to_csv("X_val_original_base.csv", index=True)  # Save for debugging

            if X_train_scaled.empty or X_val_scaled.empty or fitted_x_scaler is None or fitted_y_scaler is None:
                print("Data processing failed. Aborting.")
                return
            # Store column names needed later for saving scaler info
            feature_columns_list = X_train_scaled.columns.tolist()
            target_columns_list = y_train_original_base.columns.tolist()

            # Train the specific model
            if model_arch == "LightGBM_Baseline":
                hyperparameters = {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "random_state": 42}
                feature_config = {
                    "target": "Net_Load_Demand",
                    "scaling_X": "MinMaxScaler",
                    "scaling_y": "MinMaxScaler",
                    "output_hours": y_train_scaled.shape[1],
                }
                model_object_trained, final_validation_metrics, y_pred_val_lgbm_orig = train_lightgbm_model(
                    X_train_scaled, y_train_scaled, X_val_scaled, y_val_original, fitted_y_scaler, verbose=1
                )
                y_pred_val_plot = y_pred_val_lgbm_orig
            elif model_arch == "ANN_Baseload" or model_arch == "ANN_Change_in_Load":
                hyperparameters = {"layers": [50], "dropout": 0.2, "optimizer": "adam", "epochs": 50, "batch_size": 32, "patience": 8}
                feature_config = {
                    "target": "Net_Load_Change" if change_in_load else "Net_Load_Demand",
                    "scaling_X": "MinMaxScaler",
                    "scaling_y": "MinMaxScaler",
                    "output_hours": y_train_scaled.shape[1],
                }
                model_object_trained, final_validation_metrics, _, y_pred_val_ann_orig = train_ann_model(
                    X_train_scaled,
                    y_train_scaled,
                    X_val_scaled,
                    y_val_scaled,
                    X_train_original,
                    X_val_original,
                    y_train_original_base,
                    y_val_original,
                    fitted_y_scaler,
                    change_in_load,
                )
                y_pred_val_plot = y_pred_val_ann_orig
            elif model_arch == "LSTM_Baseload" or model_arch == "LSTM_Change_in_Load":
                hyperparameters = {"lstm_units": 50, "dropout": 0.2, "optimizer": "adam", "epochs": 50, "batch_size": 32, "patience": 8}
                feature_config = {
                    "target": "Net_Load_Change" if change_in_load else "Net_Load_Demand",
                    "scaling_X": "MinMaxScaler",
                    "scaling_y": "MinMaxScaler",
                    "output_hours": y_train_scaled.shape[1],
                }
                model_object_trained, final_validation_metrics, _, y_pred_val_lstm_orig = train_lstm_model(
                    X_train_scaled,
                    y_train_scaled,
                    X_val_scaled,
                    y_val_scaled,
                    X_train_original,
                    X_val_original,
                    y_train_original_base,
                    y_val_original,
                    fitted_y_scaler,
                    change_in_load,
                )
                y_pred_val_plot = y_pred_val_lstm_orig
            else:
                print(f"Error: Unknown single model architecture '{model_arch}'")
                return

        # =======================================================
        # === Handle Intermediate RLS Combined Stages =========
        # =======================================================
        elif is_ann_rls_run or is_lstm_rls_run:
            model_type = "ANN" if is_ann_rls_run else "LSTM"
            # Pass feeder_id and version to the stage function
            _, y_pred_rls_combined_val, y_val_rls_orig_df, rls_filter_list_stage = run_rls_combination_stage(
                train_df_raw, val_df_raw, scenario, model_type, feeder_id, version  # Pass feeder_id and version
            )
            if y_pred_rls_combined_val is None:  # Check if stage failed
                print(f"ERROR: {model_type} RLS combination stage failed. Aborting run.")
                return  # Stop this training run

            model_object_trained = rls_filter_list_stage  # Save filters
            y_val_original = y_val_rls_orig_df  # Actuals for metrics/plotting
            target_columns_list = y_val_original.columns.tolist()  # Get target columns
            # Calculate metrics
            y_val_rls_orig_np = y_val_original.values
            mae = mean_absolute_error(y_val_rls_orig_np, y_pred_rls_combined_val)
            rmse = np.sqrt(mean_squared_error(y_val_rls_orig_np, y_pred_rls_combined_val))
            denominator = np.abs(y_val_rls_orig_np) + np.abs(y_pred_rls_combined_val)
            safe_denominator = np.where(denominator == 0, 1, denominator)
            smape_values = 200 * np.abs(y_pred_rls_combined_val - y_val_rls_orig_np) / safe_denominator
            smape = np.mean(smape_values)
            final_validation_metrics = {"mae": mae, "rmse": rmse, "smape": smape}
            hyperparameters = {
                "rls_mu": model_object_trained[0].mu if model_object_trained and isinstance(model_object_trained, list) else None,
                "rls_eps": model_object_trained[0].eps if model_object_trained and isinstance(model_object_trained, list) else None,
            }
            feature_config = {
                "input_models": [f"{model_type}_Baseload", f"{model_type}_Change_in_Load"],
                "target": "Net_Load_Demand",
                "combiner": "padasip.FilterRLS",
            }
            y_pred_val_plot = y_pred_rls_combined_val

        # =======================================================
        # === Handle Final RLS Combined =========================
        # =======================================================
        elif is_final_rls_run:
            print("\n--- Getting ANN_RLS Stage Results ---")
            # Pass feeder_id and version to the stage function
            y_pred_ann_rls_train, y_pred_ann_rls_val, y_val_ann_orig_df, _ = run_rls_combination_stage(
                train_df_raw, val_df_raw, scenario, "ANN", feeder_id, version  # Pass feeder_id and version
            )
            if y_pred_ann_rls_train is None:
                print("ERROR: ANN RLS stage failed. Aborting.")
                return

            print("\n--- Getting LSTM_RLS Stage Results ---")
            # Pass feeder_id and version to the stage function
            y_pred_lstm_rls_train, y_pred_lstm_rls_val, y_val_lstm_orig_df, _ = run_rls_combination_stage(
                train_df_raw, val_df_raw, scenario, "LSTM", feeder_id, version  # Pass feeder_id and version
            )
            if y_pred_lstm_rls_train is None:
                print("ERROR: LSTM RLS stage failed. Aborting.")
                return
            print("\n--- Aligning Inputs for Final RLS Combiner Training ---")
            _, y_train_original_base, _, _ = feature_engineer_and_scale(train_df_raw, scenario, change_in_load=False, apply_scaling=False)
            common_train_index = y_train_original_base.index
            # print("************************* \n", common_train_index)
            # print("y_train_original_base.index: \n", y_train_original_base.index)
            # print("y_pred_ann_rls_train.index: \n", pd.DataFrame(y_pred_ann_rls_train).index)
            # print("y_pred_lstm_rls_train.index: \n", pd.DataFrame(y_pred_lstm_rls_train).index)
            # print("*************************")
            y_train_final_orig_np = y_train_original_base.reindex(common_train_index).values
            y_pred_ann_rls_train_aligned_np = (
                pd.DataFrame(y_pred_ann_rls_train, index=common_train_index).reindex(common_train_index).values
            )  # Use common index
            y_pred_lstm_rls_train_aligned_np = (
                pd.DataFrame(y_pred_lstm_rls_train, index=common_train_index).reindex(common_train_index).values
            )  # Use common index
            print("\n--- Training Final RLS Combiner on Training Data ---")
            final_rls_mu = 0.25
            final_rls_eps = 0.15
            final_rls_filter_list = train_padasip_rls_combiner(
                y_pred_ann_rls_train_aligned_np, y_pred_lstm_rls_train_aligned_np, y_train_final_orig_np, mu=final_rls_mu, eps=final_rls_eps
            )
            model_object_trained = final_rls_filter_list  # Save final filters
            print("\n--- Predicting Final RLS Combiner on Validation Data ---")
            common_val_index = y_val_ann_orig_df.index
            y_pred_ann_rls_val_aligned_np = pd.DataFrame(y_pred_ann_rls_val, index=common_val_index).reindex(common_val_index).values
            y_pred_lstm_rls_val_aligned_np = pd.DataFrame(y_pred_lstm_rls_val, index=common_val_index).reindex(common_val_index).values
            y_pred_final_combined_val = predict_with_padasip_rls(
                final_rls_filter_list, y_val_ann_orig_df.values, y_pred_ann_rls_val_aligned_np, y_pred_lstm_rls_val_aligned_np
            )
            print("\n--- Evaluating Final RLS Combiner on Validation Data ---")
            y_val_final_orig_np = y_val_ann_orig_df.reindex(common_val_index).values
            if np.isnan(y_pred_final_combined_val).any() or np.isinf(y_pred_final_combined_val).any():
                print("ERROR: NaN or Inf detected in FINAL combined validation predictions!")
                final_validation_metrics = {"mae": np.nan, "rmse": np.nan, "smape": np.nan}
            else:
                mae = mean_absolute_error(y_val_final_orig_np, y_pred_final_combined_val)
                rmse = np.sqrt(mean_squared_error(y_val_final_orig_np, y_pred_final_combined_val))
                denominator = np.abs(y_val_final_orig_np) + np.abs(y_pred_final_combined_val)
                safe_denominator = np.where(denominator == 0, 1, denominator)
                smape_values = 200 * np.abs(y_pred_final_combined_val - y_val_final_orig_np) / safe_denominator
                smape = np.mean(smape_values)
                final_validation_metrics = {"mae": mae, "rmse": rmse, "smape": smape}
            hyperparameters = {"final_rls_mu": final_rls_mu, "final_rls_eps": final_rls_eps}
            feature_config = {"input_models": ["ANN_RLS_Combined", "LSTM_RLS_Combined"], "target": "Net_Load_Demand", "combiner": "padasip.FilterRLS"}
            y_pred_val_plot = y_pred_final_combined_val
            y_val_original = y_val_ann_orig_df.reindex(common_val_index)  # Use aligned validation actuals
            target_columns_list = y_val_original.columns.tolist()  # Get target columns

        else:
            print(f"Error: Logic error, model architecture '{model_arch}' not handled.")
            return

        # =============================================
        # === Saving and Logging (Modified Logic) =====
        # =============================================
        if model_object_trained is None:
            print("No model object trained to save. Aborting.")
            return

        print(f"\nFinal Validation Metrics (Original Scale): {final_validation_metrics}")
        # --- Plotting (Unchanged) ---
        if y_pred_val_plot is not None and y_val_original is not None:
            try:
                plot_title_prefix = f"Validation ({model_arch})"
                print(f"Generating validation plot ({plot_title_prefix} - Original Scale)...")
                if not isinstance(y_val_original, pd.DataFrame):
                    print("Warning: y_val_original is not a DataFrame, cannot extract column names for plotting.")
                    raise TypeError("Cannot determine target columns for plotting.")
                else:
                    plot_target_columns = y_val_original.columns.tolist()
                actual_flat = y_val_original.values.flatten()
                pred_flat = y_pred_val_plot.flatten()
                actual_hours = sorted([int(col.split("_Hour_")[-1]) for col in plot_target_columns])
                num_hours = len(actual_hours)
                num_days = len(y_val_original)
                base_dates = pd.to_datetime(np.repeat(y_val_original.index, num_hours))
                hour_offsets = pd.to_timedelta(np.tile(actual_hours, num_days), unit="h")
                plot_index = base_dates + hour_offsets
                if len(actual_flat) != len(pred_flat) or len(actual_flat) != len(plot_index):
                    min_len = min(len(actual_flat), len(pred_flat), len(plot_index))
                    print(f"Warning: Length mismatch plotting. Truncating to {min_len}")
                    actual_flat, pred_flat, plot_index = actual_flat[:min_len], pred_flat[:min_len], plot_index[:min_len]
                results_df = pd.DataFrame({"Actual": actual_flat, "Predicted": pred_flat}, index=plot_index)
                results_df = results_df.sort_index()
                print(f"Sample of {plot_title_prefix} Actual vs Predicted (Original Scale):")
                print(results_df.head(min(3, len(results_df))))
                # print(results_df.describe())
                fig = px.line(results_df, title=f"{plot_title_prefix}: Feeder {feeder_id} - {model_arch} ({scenario})")
                # fig.show()
            except Exception as plot_err:
                print(f"Could not generate validation plot: {plot_err}")
                traceback.print_exc()
        else:
            print("Skipping plotting: Prediction or actual data not available.")

        # --- Save Artifact(s) ---
        artifact_path_for_db = None  # This will hold the single path or JSON string
        try:
            if is_keras_model:
                print("Saving Keras model natively and scalers separately...")
                # 1. Save Keras model
                keras_filename = f"{model_arch}_{scenario}_{version}.keras"

                print("TESTING PURPOSE: \n")
                print(type(model_object_trained))
                print(keras_filename)
                print("TESTING PURPOSE: \n")

                keras_local_path = os.path.join(TEMP_DIR, keras_filename)
                model_object_trained.save(keras_local_path)  # Use Keras native save
                keras_storage_path = f"models/feeder_{feeder_id}/{keras_filename}"
                with open(keras_local_path, "rb") as f:
                    supabase.storage.from_(STORAGE_BUCKET).upload(path=keras_storage_path, file=f, file_options={"upsert": "true"})
                os.remove(keras_local_path)
                print(f"Keras model saved to: {keras_storage_path}")

                # 2. Save Scalers and Columns via Pickle
                scaler_info = {
                    "x_scaler": fitted_x_scaler,
                    "y_scaler": fitted_y_scaler,
                    "feature_columns": feature_columns_list,
                    "target_columns": target_columns_list,
                }
                # Use a distinct version tag for the scaler file
                scaler_version_tag = f"{version}_scalers"
                scaler_storage_path = save_pickle_artifact(scaler_info, feeder_id, model_arch, scenario, scaler_version_tag)
                print(f"Scalers/Columns saved to: {scaler_storage_path}")

                # 3. Store both paths as JSON in the DB path field
                artifact_path_for_db = json.dumps({"keras_model": keras_storage_path, "scalers_pkl": scaler_storage_path})

            else:
                # For non-Keras (LGBM, RLS filters), save bundled object using pickle
                print("Saving non-Keras model/filters and potentially scalers via pickle...")
                # RLS stages already have model_object_trained as the filters list
                if is_ann_rls_run or is_lstm_rls_run or is_final_rls_run:
                    artifact_to_save = {"rls_filters": model_object_trained, "target_columns": target_columns_list}
                    # Scalers are not directly used by RLS filters, don't save them here
                elif model_arch == "LightGBM_Baseline":
                    artifact_to_save = {
                        "model": model_object_trained,
                        "x_scaler": fitted_x_scaler,
                        "y_scaler": fitted_y_scaler,
                        "feature_columns": feature_columns_list,
                        "target_columns": target_columns_list,
                    }
                else:
                    print(f"Warning: Unhandled non-Keras architecture for saving: {model_arch}")
                    artifact_to_save = {"model": model_object_trained}  # Save model only as fallback

                # Use the main version tag for the single pickle file
                artifact_path_for_db = save_pickle_artifact(artifact_to_save, feeder_id, model_arch, scenario, version)

        except Exception as e:
            print(f"Error saving artifact(s): {e}")
            traceback.print_exc()
            return  # Don't log metadata if saving failed

        # --- Log Metadata ---
        if artifact_path_for_db is None:
            print("Error: Artifact path for database is null. Cannot log metadata.")
            return

        metadata = {
            "feeder_id": feeder_id,
            "model_architecture_type": model_arch,
            "scenario_type": scenario,
            "model_version": version,
            "train_data_start_timestamp": train_start,
            "train_data_end_timestamp": train_end,
            "model_hyperparameters": json.dumps(hyperparameters),
            "feature_engineering_config": json.dumps(feature_config),
            "model_artifact_path": artifact_path_for_db,  # Store single path or JSON string
            "validation_metrics": json.dumps(final_validation_metrics),
            "is_active_for_forecast": False,
        }

        logged_model_id = None

        try:
            logged_model_id = log_model_metadata(metadata)

            if logged_model_id is None:
                print("Warning: Failed to retrieve model_id after logging metadata. Cannot store validation results.")
        except Exception as e:
            print(f"Error logging metadata: {e}")
            # Don't proceed to store validation if metadata failed
            logged_model_id = None

        # --- Store Validation Results (NEW STEP) ---
        if logged_model_id is not None and y_pred_val_plot is not None and y_val_original is not None and target_columns_list is not None:
            print("\n--- Storing Validation Results ---")
            validation_ts = datetime.now(timezone.utc).isoformat()
            try:
                store_validation_results(
                    model_id=logged_model_id,
                    feeder_id=feeder_id,
                    validation_run_timestamp=validation_ts,
                    y_val_actual_df=y_val_original,
                    y_pred_val_original_np=y_pred_val_plot,
                    target_columns=target_columns_list,
                )
            except Exception as e:
                # Log error but don't necessarily fail the whole training run
                print(f"ERROR occurred while storing validation results: {e}")
                traceback.print_exc()
        elif logged_model_id is None:
            print("Skipping validation result storage because model metadata logging failed.")
        else:
            print("Skipping validation result storage because prediction/actual data/target columns are missing.")

    except Exception as e:
        print(f"An error occurred during the training run for {model_arch}: {e}")
        traceback.print_exc()
    finally:
        supabase.postgrest.schema("public")  # Reset schema

    print(f"--- Training Run Completed for Feeder {feeder_id}, {model_arch}, {scenario} ---")
