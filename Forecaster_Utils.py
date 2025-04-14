try:
    import tensorflow as tf
    from keras.models import Sequential, save_model, load_model
    from keras.layers import Dense, LSTM, Input, Dropout
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
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from DB_Utils import select_model_for_forecast, load_artifact_from_storage, store_forecasts, fetch_data, NormalizeLayer
    from supabase import create_client, Client
    import traceback
    import plotly.express as px
    from dotenv import load_dotenv, find_dotenv

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Or '2' or '3'

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


# def fetch_data(feeder_id, start_date, end_date):
#     """Fetches combined feeder and weather data from Supabase."""
#     print(f"Fetching data for Feeder {feeder_id} from {start_date} to {end_date}...")
#     try:
#         supabase.postgrest.schema(DATA_SCHEMA)
#         response = (
#             supabase.table(f"Feeder_Weather_Combined_Data")
#             .select("*")
#             .eq("Feeder_ID", feeder_id)
#             .gte("Timestamp", start_date)
#             .lt("Timestamp", end_date)
#             .order("Timestamp", desc=False)
#             .execute()
#         )
#         if not response.data:
#             print(f"Warning: No data found for Feeder {feeder_id} in the specified range.")
#             return pd.DataFrame()
#         df = pd.DataFrame(response.data)
#         df["Timestamp"] = pd.to_datetime(df["Timestamp"])
#         df = df.set_index("Timestamp")
#         print(f"Fetched {len(df)} records.")
#         return df
#     except Exception as e:
#         print(f"Error fetching data: {e}")
#         raise


def prepare_daily_vectors(df, feature_cols, target_col_list, scenario_hours):
    """Pivots hourly data into daily vectors (one row per day)."""
    print("Pivoting data into daily vectors...")
    df_copy = df.copy()
    df_copy["date"] = df_copy.index.date
    df_copy["hour"] = df_copy.index.hour
    if scenario_hours:
        df_copy = df_copy[df_copy["hour"].isin(scenario_hours)]
    pivoted_X = df_copy.pivot_table(index="date", columns="hour", values=feature_cols)
    pivoted_X.columns = [f"{col[0]}_Hour_{col[1]}" for col in pivoted_X.columns]
    try:
        pivoted_y = df_copy.pivot_table(index="date", columns="hour", values=target_col_list)
        if len(target_col_list) == 1:
            if isinstance(pivoted_y, pd.Series):
                pivoted_y = pivoted_y.to_frame()
            pivoted_y.columns = [f"{target_col_list[0]}_Hour_{col[1]}" for col in pivoted_y.columns]
        else:
            pivoted_y.columns = [f"{col[0]}_Hour_{col[1]}" for col in pivoted_y.columns]
    except Exception as e:
        print(f"Warning: Could not pivot target columns {target_col_list} (may not exist in forecast input): {e}")
        pivoted_y = pd.DataFrame(index=pivoted_X.index)
    expected_hours = scenario_hours if scenario_hours else list(range(24))
    ordered_X_columns = [f"{feat}_Hour_{hr}" for feat in feature_cols for hr in expected_hours]
    pivoted_X = pivoted_X.reindex(columns=ordered_X_columns)
    pivoted_X.index = pd.to_datetime(pivoted_X.index)
    pivoted_X["DOW"] = pivoted_X.index.dayofweek
    pivoted_X["Is_Holiday"] = 0
    pivoted_X = pd.get_dummies(pivoted_X, columns=["DOW"], prefix="DOW", dtype="int")

    # --- Ensure all DOW columns exist ---
    expected_dow_cols = [f"DOW_{i}" for i in range(7)]
    for col in expected_dow_cols:
        if col not in pivoted_X.columns:
            print(f"Adding missing column: {col}")
            pivoted_X[col] = 0  # Add missing DOW columns and set to 0
    # Ensure correct dtype after adding potentially missing columns
    pivoted_X[expected_dow_cols] = pivoted_X[expected_dow_cols].astype(int)

    pivoted_X["Is_Weekend"] = pivoted_X["DOW_5"] + pivoted_X["DOW_6"]

    DOW_ordered_columns = [f"DOW_{i}" for i in range(7)]
    pivoted_X = pivoted_X.reindex(columns=ordered_X_columns + DOW_ordered_columns + ["Is_Holiday", "Is_Weekend"])

    pivoted_X.index = pd.to_datetime(pivoted_X.index)
    pivoted_y.index = pd.to_datetime(pivoted_y.index)

    # print(pivoted_X.columns)

    print(f"Reshaped data: X shape {pivoted_X.shape}")
    return pivoted_X, pivoted_y


def feature_engineer_and_scale(df, scenario, target_date=None, x_scaler=None, y_scaler=None, change_in_load=False, apply_scaling=True):
    """Prepares features, reshapes data, applies MinMaxScaler to X and y."""
    print(f"Starting feature engineering for scenario: {scenario}...")
    df_processed = df.copy()
    df_processed["Net_Load_Change"] = df_processed["Net_Load_Demand"].diff(24).fillna(0)
    df_processed["Prev_Day_Net_Load_Demand"] = df_processed["Net_Load_Demand"].shift(24)
    df_processed["Prev_Day_Temperature_Historic"] = df_processed["Temperature_Historic"].shift(24)
    df_processed["Prev_Day_Shortwave_Radiation_Historic"] = df_processed["Shortwave_Radiation_Historic"].shift(24)
    df_processed["Prev_Day_Cloud_Cover_Historic"] = df_processed["Cloud_Cover_Historic"].shift(24)

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
    scenario_hours = DAY_HOURS if scenario == "Day" else NIGHT_HOURS if scenario == "Night" else None

    df_processed = df_processed.dropna()

    X, y = prepare_daily_vectors(df_processed, feature_cols, target_col, scenario_hours)

    # --- Filter for Target Date ---
    # Do this *after* prepare_daily_vectors has created DOW columns
    if target_date:
        X = X[X.index.date == target_date]

    if X.empty:
        print(f"Warning: No data row found for target date {target_date} after reshaping.")
        # Return empty dataframe matching expected structure if possible, else None
        # This depends on whether feature_columns are known at this point
        return pd.DataFrame(), None, None, None

    # print("+++++++++++++++++++++++++")
    # print(y.index)
    # print("+++++++++++++++++++++++++")

    if X.empty:
        print(f"Warning: No data row found for target date after reshaping.")
        return X, None, None, None
    if X.isnull().values.any():
        print("ERROR: NaNs detected in the final input vector for forecasting!")
        print(X[X.isnull().any(axis=1)])
        raise ValueError("NaNs found in input features for forecast day. Check data fetching and feature engineering.")
    if not apply_scaling:
        print("Scaling is disabled.")
        return X, y, None, None
    if x_scaler is None:
        raise ValueError("x_scaler must be provided for forecasting mode.")
    print("Transforming input features (X) using provided scaler...")
    if not hasattr(x_scaler, "transform"):
        raise ValueError("Provided x_scaler object must have a 'transform' method.")
    try:
        if hasattr(x_scaler, "feature_names_in_") and list(X.columns) != list(x_scaler.feature_names_in_):
            print("Warning: Feature mismatch detected. Attempting reorder...")
            X = X[x_scaler.feature_names_in_]
        elif hasattr(x_scaler, "n_features_in_") and X.shape[1] != x_scaler.n_features_in_:
            raise ValueError(f"Input feature count mismatch: data has {X.shape[1]}, X scaler expects {x_scaler.n_features_in_}")
        X_scaled = x_scaler.transform(X)
        print("Input features transformed.")
        X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        return X_scaled_df, y, None, None
    except Exception as e:
        print(f"Error applying scaler transform: {e}")
        traceback.print_exc()
        raise


def convert_change_in_load_to_base_load(X_original, y_pred_change_original):
    """Converts predicted change_in_load back to base_load."""
    X_original_np = X_original.values if isinstance(X_original, pd.DataFrame) else X_original
    y_pred_change_np = y_pred_change_original.values if isinstance(y_pred_change_original, pd.DataFrame) else y_pred_change_original
    prev_day_cols = [col for col in X_original.columns if col.startswith("Prev_Day_Net_Load_Demand_Hour_")]
    if len(prev_day_cols) == 0:
        raise ValueError("Could not find 'Prev_Day_Net_Load_Demand_Hour_' columns in X_original.")
    if len(prev_day_cols) != y_pred_change_np.shape[1]:
        raise ValueError(
            f"Mismatch between number of previous day load columns ({len(prev_day_cols)}) and prediction columns ({y_pred_change_np.shape[1]})"
        )
    prev_day_indices = [X_original.columns.get_loc(col) for col in prev_day_cols]
    prev_day_load = X_original_np[:, prev_day_indices]
    y_pred_base_np = prev_day_load.astype(float) + y_pred_change_np.astype(float)
    print("Converted change_in_load prediction back to base_load prediction.")
    return y_pred_base_np


# def predict_with_padasip_rls(rls_filters, actuals, predictions1, predictions2):
#     # print("+++++++++++++++++++++++++++")
#     # print(actuals)
#     # print(predictions1)
#     # print(predictions2)
#     # print("+++++++++++++++++++++++++++")

#     """Combines predictions using a list of fitted padasip RLS filters."""
#     n_samples, n_outputs = predictions1.shape
#     if len(rls_filters) != n_outputs:
#         raise ValueError("Number of RLS filters does not match number of prediction outputs.")
#     combined_predictions = np.zeros_like(predictions1)
#     for t in range(n_samples):
#         for k in range(n_outputs):
#             x_k = np.array([predictions1[t, k], predictions2[t, k]])
#             d_k = actuals[t, k]
#             combined_predictions[t, k] = rls_filters[k].predict(x_k)

#             rls_filters[k].adapt(d_k, x_k)

#             # print("+++++++++++++++++++++++++++")
#             # print("Input, Predicted output: x_k, combined_predictions: \n", x_k, combined_predictions[t, k], end=" ")
#             # print(" \n", )
#             # print("+++++++++++++++++++++++++++")
#     return combined_predictions

# Forecaster_Utils.py


def predict_with_padasip_rls(rls_filters, y_scaler, actuals_orig, predictions1_orig, predictions2_orig):
    """
    Combines predictions using fitted RLS filters, performing scaling internally.
    Adapts the filters based on provided actuals.

    Args:
        rls_filters (list): List of fitted padasip FilterRLS objects.
        y_scaler (object): Fitted MinMaxScaler object for the target variable.
        actuals_orig (np.array): Actual target values in original scale.
        predictions1_orig (np.array): Predictions from model 1 in original scale.
        predictions2_orig (np.array): Predictions from model 2 in original scale.

    Returns:
        np.array: Combined predictions in the original scale.
    """
    if not PADASIP_AVAILABLE:
        raise RuntimeError("padasip library not found, cannot use RLS filters.")
    if y_scaler is None or not hasattr(y_scaler, "transform") or not hasattr(y_scaler, "inverse_transform"):
        raise ValueError("A valid, fitted y_scaler (MinMaxScaler) must be provided.")

    # Input validation
    if actuals_orig is None:
        raise ValueError("Actual values are required for RLS prediction with adaptation.")
    if predictions1_orig.shape != predictions2_orig.shape or predictions1_orig.shape != actuals_orig.shape:
        raise ValueError(f"Shape mismatch: actuals {actuals_orig.shape}, preds1 {predictions1_orig.shape}, preds2 {predictions2_orig.shape}")

    n_samples, n_outputs = predictions1_orig.shape
    if len(rls_filters) != n_outputs:
        raise ValueError("Number of RLS filters does not match number of prediction outputs.")

    # --- Scale Inputs and Actuals ---
    print("Scaling inputs and actuals for RLS predict/adapt...")
    try:
        # Ensure shapes are correct for scaler (samples, features/hours)
        if actuals_orig.shape[1] != y_scaler.n_features_in_:
            raise ValueError("Actuals shape mismatch with y_scaler")
        if predictions1_orig.shape[1] != y_scaler.n_features_in_:
            raise ValueError("Predictions1 shape mismatch with y_scaler")
        if predictions2_orig.shape[1] != y_scaler.n_features_in_:
            raise ValueError("Predictions2 shape mismatch with y_scaler")

        actuals_scaled = y_scaler.transform(actuals_orig)
        predictions1_scaled = y_scaler.transform(predictions1_orig)
        predictions2_scaled = y_scaler.transform(predictions2_orig)
    except Exception as scale_err:
        print(f"ERROR during scaling for RLS: {scale_err}")
        traceback.print_exc()
        raise ValueError("Failed to scale data for RLS.") from scale_err
    # --- End Scaling ---

    combined_predictions_scaled = np.zeros_like(predictions1_scaled)

    print(f"Running RLS predict-adapt loop on SCALED data for {n_samples} sample(s)...")
    for t in range(n_samples):  # Loop through samples
        for k in range(n_outputs):  # Loop through hours/outputs
            x_k_scaled = np.array([predictions1_scaled[t, k], predictions2_scaled[t, k]])
            d_k_scaled = actuals_scaled[t, k]  # Scaled actual value

            # --- Prediction Step (using scaled inputs) ---
            if np.isnan(x_k_scaled).any() or np.isinf(x_k_scaled).any():
                print(f"Warning: NaN/Inf SCALED input for RLS predict at sample {t}, hour {k}. Setting scaled output to NaN.")
                combined_predictions_scaled[t, k] = np.nan
                continue
            try:
                combined_predictions_scaled[t, k] = rls_filters[k].predict(x_k_scaled)
            except Exception as predict_err:
                print(f"ERROR during RLS predict (scaled) at sample {t}, hour {k}: {predict_err}")
                combined_predictions_scaled[t, k] = np.nan
                continue

            # --- Adaptation Step (using scaled inputs and target) ---
            if np.isnan(d_k_scaled) or np.isinf(d_k_scaled):
                print(f"Warning: NaN/Inf SCALED target value at sample {t}, hour {k}. Skipping RLS adapt.")
                continue
            try:
                rls_filters[k].adapt(d_k_scaled, x_k_scaled)
            except Exception as adapt_err:
                print(f"ERROR during RLS adapt (scaled) at sample {t}, hour {k}: {adapt_err}")
                pass

    print("RLS predict-adapt loop finished.")

    # --- Inverse Transform the Combined Prediction ---
    print("Inverse transforming combined RLS prediction...")
    # Handle potential NaNs introduced during predict/adapt before inverse transform
    if np.isnan(combined_predictions_scaled).any():
        print("Warning: NaNs detected in scaled RLS predictions before inverse transform.")
        # Option 1: Keep NaNs - inverse_transform might handle them or raise error depending on sklearn version
        # Option 2: Impute NaNs (e.g., with mean of non-NaN scaled preds) - complex
        # Let's proceed and let inverse_transform handle it for now.
    try:
        combined_predictions_original = y_scaler.inverse_transform(combined_predictions_scaled)
    except Exception as inv_err:
        print(f"ERROR during inverse transform of RLS predictions: {inv_err}")
        traceback.print_exc()
        # Return scaled preds or NaNs if inverse fails? Returning NaNs is safer.
        combined_predictions_original = np.full_like(combined_predictions_scaled, np.nan)
    # --- End Inverse Transform ---

    return combined_predictions_original


def run_forecast(feeder_id, target_date_str, architecture, scenario, version=None):  # Changed version_prefix to version
    """Orchestrates the forecasting process by calling get_prediction with a specific version."""
    global _prediction_cache
    _prediction_cache = {}  # Clear cache for each new run_forecast call

    print(f"\n--- Starting Forecast Run ---")
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    print(f"Feeder: {feeder_id}, Target Date: {target_date}, Arch: {architecture}, Scenario: {scenario}, Version: {version}")
    forecast_run_timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # Get the final prediction using the recursive function
        # Pass the specific version requested (can be None to get latest)
        final_prediction_original, target_columns, final_model_id, y_scaler_to_use = get_prediction(
            feeder_id, target_date, architecture, scenario, version=version  # Pass specific version
        )

        if final_prediction_original is None:
            print("ERROR: Failed to obtain final prediction.")
            return
        if target_columns is None:
            print("ERROR: Target column names could not be determined.")
            return

        # Store the final result using the ID of the model ultimately selected/used
        store_forecasts(final_model_id, feeder_id, forecast_run_timestamp, target_date, final_prediction_original, target_columns)

        print(f"--- Forecast Run Successfully Completed for {architecture}/{scenario} (Version: {version}) ---")

    except Exception as e:
        print(f"--- Forecast Run Failed for {architecture}/{scenario} (Version: {version}) ---")
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        supabase.postgrest.schema("public")  # Reset schema
        _prediction_cache = {}  # Clear cache


# def get_prediction(feeder_id, target_date, architecture, scenario, version=None):  # Changed version_prefix to version
#     """
#     Gets the final (original scale) prediction for a given model, handling recursion for RLS.
#     Uses a cache to avoid re-computing predictions for the same model/date/version.
#     Requires input models for RLS stages to have the SAME version string.
#     """
#     global _prediction_cache
#     # Use version in cache key
#     cache_key = (feeder_id, target_date, architecture, scenario, version)
#     if cache_key in _prediction_cache:
#         print(f"Cache HIT for: {cache_key}")
#         return _prediction_cache[cache_key]
#     else:
#         print(f"Cache MISS for: {cache_key}. Computing prediction...")

#     # 1. Select Model Metadata (using exact version if provided)
#     model_metadata = select_model_for_forecast(feeder_id, architecture, scenario, version=version)  # Pass exact version
#     if not model_metadata:
#         # If version was specified but not found, raise error
#         if version:
#             raise ValueError(f"Could not find model metadata for specific version {version} and criteria {cache_key}")
#         # If version was None, select_model already tried latest, so still raise error
#         else:
#             raise ValueError(f"Could not find any model metadata for {cache_key}")

#     actual_model_id = model_metadata["model_id"]
#     artifact_path_info = model_metadata["model_artifact_path"]
#     model_arch = model_metadata["model_architecture_type"]
#     model_scenario = model_metadata["scenario_type"]
#     # Use the actual version found in metadata (important if version was initially None)
#     actual_version_used = model_metadata["model_version"]
#     feature_config = model_metadata.get("feature_engineering_config", {})
#     predicts_change = feature_config.get("target") == "Net_Load_Change" or "Change_in_Load" in model_arch

#     # 2. Load Artifact(s)
#     loaded_artifact = load_artifact_from_storage(artifact_path_info)

#     # 3. Check Artifact Type and Predict
#     final_prediction_original = None
#     target_columns_final = loaded_artifact.get("target_columns")

#     # --- Case 1: RLS Filters Artifact ---
#     if "rls_filters" in loaded_artifact and loaded_artifact.get("rls_filters") is not None:
#         print(f"Processing RLS artifact for {model_arch} (Version: {actual_version_used})...")
#         rls_filters = loaded_artifact["rls_filters"]
#         if not PADASIP_AVAILABLE:
#             raise RuntimeError("padasip library not found, cannot use RLS filters.")

#         input_models = feature_config.get("input_models")
#         if not input_models or len(input_models) != 2:
#             raise ValueError(f"Invalid or missing 'input_models' in feature_config for RLS model {model_arch}")

#         print(f"RLS requires inputs from: {input_models}")
#         # Recursively get predictions for input models using the SAME EXACT version
#         input_pred_1_result = get_prediction(feeder_id, target_date, input_models[0], scenario, actual_version_used)
#         input_pred_2_result = get_prediction(feeder_id, target_date, input_models[1], scenario, actual_version_used)

#         if input_pred_1_result is None or input_pred_2_result is None:
#             raise RuntimeError(f"Failed to get predictions for one or both input models for {model_arch} (Version: {actual_version_used})")

#         # Extract predictions (first element of the tuple)
#         input_pred_1 = input_pred_1_result[0]
#         input_pred_2 = input_pred_2_result[0]

#         print(f"Combining predictions for {model_arch} using RLS filters...")
#         final_prediction_original = predict_with_padasip_rls(rls_filters, input_pred_1, input_pred_2)
#         target_columns_final = loaded_artifact.get("target_columns")  # From RLS artifact

#     # --- Case 2: Base Model Artifact (LGBM, Keras) ---
#     elif "model" in loaded_artifact and loaded_artifact.get("model") is not None:
#         print(f"Processing base model artifact for {model_arch} (Version: {actual_version_used})...")
#         model = loaded_artifact["model"]
#         x_scaler = loaded_artifact["x_scaler"]
#         y_scaler = loaded_artifact["y_scaler"]
#         feature_columns = loaded_artifact["feature_columns"]
#         target_columns_final = loaded_artifact.get("target_columns")  # From base artifact

#         if model is None or x_scaler is None or y_scaler is None or feature_columns is None or target_columns_final is None:
#             raise ValueError(f"Loaded artifact for base model {model_arch} (Version: {actual_version_used}) is missing required components.")

#         # Fetch data needed
#         start_fetch_dt = datetime.combine(target_date - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
#         end_fetch_dt = datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
#         input_df_raw = fetch_data(feeder_id, start_fetch_dt.isoformat(), end_fetch_dt.isoformat())
#         if input_df_raw.empty:
#             raise ValueError(f"Insufficient data fetched for {model_arch} prediction.")
#         target_day_data = input_df_raw[input_df_raw.index.date == target_date]
#         required_forecast_cols = ["Temperature_Forecast", "Shortwave_Radiation_Forecast", "Cloud_Cover_Forecast"]
#         if target_day_data.empty or target_day_data[required_forecast_cols].isnull().values.any():
#             raise ValueError(f"Missing weather forecast data for {target_date}.")

#         # Prepare input features
#         X_scaled_df, _, _, _ = feature_engineer_and_scale(
#             input_df_raw, model_scenario, target_date, x_scaler=x_scaler, y_scaler=y_scaler, change_in_load=predicts_change, apply_scaling=True
#         )
#         X_scaled_target_day = X_scaled_df
#         if X_scaled_target_day.empty:
#             raise ValueError(f"No input feature vector generated for {model_arch}.")
#         if list(X_scaled_target_day.columns) != feature_columns:
#             print(f"Warning: Feature columns mismatch for {model_arch}. Attempting reorder.")
#             try:
#                 X_scaled_target_day = X_scaled_target_day[feature_columns]
#             except KeyError as ke:
#                 missing = set(feature_columns) - set(X_scaled_target_day.columns)
#                 extra = set(X_scaled_target_day.columns) - set(feature_columns)
#                 raise ValueError(f"Feature mismatch for {model_arch}. Missing: {missing}. Extra: {extra}") from ke

#         # Predict scaled values
#         print(f"Generating scaled predictions for {model_arch}...")
#         is_lstm = KERAS_AVAILABLE and isinstance(model, tf.keras.Model) and any(isinstance(layer, tf.keras.layers.LSTM) for layer in model.layers)
#         if is_lstm:
#             X_input_final = X_scaled_target_day.values.reshape((1, 1, X_scaled_target_day.shape[1]))
#         else:
#             X_input_final = X_scaled_target_day.values
#         y_pred_scaled = model.predict(X_input_final)

#         # Inverse transform
#         print(f"Inverse transforming predictions for {model_arch}...")
#         if y_pred_scaled.shape[1] != y_scaler.n_features_in_:
#             raise ValueError(f"Prediction shape mismatch for inverse transform ({model_arch}).")
#         y_pred_original = y_scaler.inverse_transform(y_pred_scaled)

#         # Post-process
#         final_prediction_original = y_pred_original
#         if predicts_change:
#             print(f"Converting change_in_load prediction for {model_arch}...")
#             X_original_target_day, _, _, _ = feature_engineer_and_scale(
#                 input_df_raw, model_scenario, target_date, x_scaler=x_scaler, y_scaler=y_scaler, change_in_load=predicts_change, apply_scaling=False
#             )
#             if X_original_target_day.empty:
#                 raise ValueError(f"Could not retrieve original X data for change conversion ({model_arch}).")
#             final_prediction_original = convert_change_in_load_to_base_load(X_original_target_day, y_pred_original)

#     # --- Case 3: Unknown Artifact Type ---
#     else:
#         raise TypeError(f"Loaded artifact for {model_arch} (Version: {actual_version_used}) has unknown structure.")

#     # --- Store result in cache and return ---
#     if final_prediction_original is None:
#         raise ValueError(f"Failed to generate final prediction for {cache_key}")

#     # Return prediction, target columns, and the specific model ID used
#     result = (final_prediction_original, target_columns_final, actual_model_id)
#     _prediction_cache[cache_key] = result
#     print(f"Prediction computed and cached for: {cache_key}")
#     return result


# Global cache for predictions within a single run_forecast execution
_prediction_cache = {}


# def get_prediction(feeder_id, target_date, architecture, scenario, version=None):
#     """
#     Gets the final (original scale) prediction for a given model, handling recursion for RLS
#     and fetching actuals needed for RLS adaptation during prediction.
#     Uses a cache to avoid re-computing predictions for the same model/date/version.
#     Requires input models for RLS stages to have the SAME version string.
#     """
#     global _prediction_cache
#     cache_key = (feeder_id, target_date, architecture, scenario, version)
#     if cache_key in _prediction_cache:
#         print(f"Cache HIT for: {cache_key}")
#         return _prediction_cache[cache_key]
#     else:
#         print(f"Cache MISS for: {cache_key}. Computing prediction...")

#     # 1. Select Model Metadata
#     model_metadata = select_model_for_forecast(feeder_id, architecture, scenario, version=version)
#     if not model_metadata:
#         if version:
#             raise ValueError(f"Could not find model metadata for specific version {version} and criteria {cache_key}")
#         else:
#             raise ValueError(f"Could not find any model metadata for {cache_key}")

#     actual_model_id = model_metadata["model_id"]
#     artifact_path_info = model_metadata["model_artifact_path"]
#     model_arch = model_metadata["model_architecture_type"]
#     model_scenario = model_metadata["scenario_type"]
#     actual_version_used = model_metadata["model_version"]
#     feature_config = model_metadata.get("feature_engineering_config", {})
#     predicts_change = feature_config.get("target") == "Net_Load_Change" or "Change_in_Load" in model_arch

#     # 2. Load Artifact(s)
#     loaded_artifact = load_artifact_from_storage(artifact_path_info)

#     # 3. Check Artifact Type and Predict
#     final_prediction_original = None
#     target_columns_final = loaded_artifact.get("target_columns")

#     # --- Case 1: RLS Filters Artifact ---
#     if "rls_filters" in loaded_artifact and loaded_artifact.get("rls_filters") is not None:
#         print(f"Processing RLS artifact for {model_arch} (Version: {actual_version_used})...")
#         rls_filters = loaded_artifact["rls_filters"]
#         target_columns_final = loaded_artifact.get("target_columns")  # Get target columns from RLS artifact
#         if not target_columns_final:
#             raise ValueError("RLS artifact missing 'target_columns'.")

#         # --- Fetch Actual Data Needed for RLS Adaptation ---
#         # RLS adaptation needs the actual base load for the target date
#         print(f"Fetching actual base load data for RLS adaptation (Target Date: {target_date})...")
#         # Fetch data covering the target date (and potentially day before for feature eng context)
#         start_fetch_dt_actuals = datetime.combine(target_date - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
#         end_fetch_dt_actuals = datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
#         actuals_df_raw = fetch_data(feeder_id, start_fetch_dt_actuals.isoformat(), end_fetch_dt_actuals.isoformat())
#         if actuals_df_raw.empty:
#             raise ValueError(f"Insufficient data fetched to get actuals for RLS adaptation on {target_date}.")

#         # Use feature_engineer_and_scale to get the correctly shaped *original* target data
#         # We don't need scalers here, just the original y dataframe/array
#         # Use change_in_load=False because RLS combines to predict base load
#         _, y_actual_original_df, _, _ = feature_engineer_and_scale(
#             actuals_df_raw,
#             model_scenario,
#             target_date,
#             x_scaler=None,
#             y_scaler=None,  # No scalers needed
#             change_in_load=False,  # We need the base load actuals
#             apply_scaling=False,  # Do not scale
#         )

#         print(actuals_df_raw, y_actual_original_df)

#         # Filter for the target date and ensure alignment
#         y_actual_original_target_day = y_actual_original_df[y_actual_original_df.index.date == target_date]

#         if y_actual_original_target_day.empty:
#             raise ValueError(f"Could not extract actual target values for {target_date} for RLS adaptation.")
#         if len(y_actual_original_target_day) > 1:
#             print(f"Warning: Multiple actual rows found for target date {target_date}. Using first.")
#             y_actual_original_target_day = y_actual_original_target_day.head(1)

#         # Ensure columns match the target_columns from the artifact
#         if list(y_actual_original_target_day.columns) != target_columns_final:
#             print("Warning: Actuals columns mismatch RLS target columns. Attempting reorder.")
#             try:
#                 y_actual_original_target_day = y_actual_original_target_day[target_columns_final]
#             except KeyError as ke:
#                 missing = set(target_columns_final) - set(y_actual_original_target_day.columns)
#                 extra = set(y_actual_original_target_day.columns) - set(target_columns_final)
#                 raise ValueError(f"Actuals column mismatch for RLS. Missing: {missing}. Extra: {extra}") from ke

#         # Convert actuals to numpy array for predict_with_padasip_rls
#         actuals_np = y_actual_original_target_day.values
#         # --- End Fetch Actual Data ---

#         # Identify input models from config
#         input_models = feature_config.get("input_models")
#         if not input_models or len(input_models) != 2:
#             raise ValueError(f"Invalid or missing 'input_models' in feature_config for RLS model {model_arch}")

#         print(f"RLS requires inputs from: {input_models}")
#         # Recursively get predictions for input models using the SAME EXACT version
#         input_pred_1_result = get_prediction(feeder_id, target_date, input_models[0], scenario, actual_version_used)
#         input_pred_2_result = get_prediction(feeder_id, target_date, input_models[1], scenario, actual_version_used)

#         if input_pred_1_result is None or input_pred_2_result is None:
#             raise RuntimeError(f"Failed to get predictions for one or both input models for {model_arch} (Version: {actual_version_used})")

#         input_pred_1 = input_pred_1_result[0]  # Prediction array
#         input_pred_2 = input_pred_2_result[0]  # Prediction array

#         # Check shape consistency before combining
#         if input_pred_1.shape != actuals_np.shape or input_pred_2.shape != actuals_np.shape:
#             raise ValueError(
#                 f"Shape mismatch before RLS combination: Actuals={actuals_np.shape}, Pred1={input_pred_1.shape}, Pred2={input_pred_2.shape}"
#             )

#         # Combine predictions using loaded RLS filters AND provide actuals for adaptation
#         print(f"Combining predictions for {model_arch} using RLS filters (with adaptation)...")
#         final_prediction_original = predict_with_padasip_rls(rls_filters, actuals_np, input_pred_1, input_pred_2)  # Pass the fetched actuals
#         # target_columns_final already set from RLS artifact

#     # --- Case 2: Base Model Artifact (LGBM, Keras) ---
#     elif "model" in loaded_artifact and loaded_artifact.get("model") is not None:
#         # --- This block remains the same as before ---
#         print(f"Processing base model artifact for {model_arch} (Version: {actual_version_used})...")
#         model = loaded_artifact["model"]
#         x_scaler = loaded_artifact["x_scaler"]
#         y_scaler = loaded_artifact["y_scaler"]
#         feature_columns = loaded_artifact["feature_columns"]
#         target_columns_final = loaded_artifact.get("target_columns")
#         if model is None or x_scaler is None or y_scaler is None or feature_columns is None or target_columns_final is None:
#             raise ValueError(f"Loaded artifact for base model {model_arch} (Version: {actual_version_used}) is missing required components.")
#         start_fetch_dt = datetime.combine(target_date - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
#         end_fetch_dt = datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
#         input_df_raw = fetch_data(feeder_id, start_fetch_dt.isoformat(), end_fetch_dt.isoformat())
#         if input_df_raw.empty:
#             raise ValueError(f"Insufficient data fetched for {model_arch} prediction.")
#         target_day_data = input_df_raw[input_df_raw.index.date == target_date]
#         required_forecast_cols = ["Temperature_Forecast", "Shortwave_Radiation_Forecast", "Cloud_Cover_Forecast"]
#         if target_day_data.empty or target_day_data[required_forecast_cols].isnull().values.any():
#             raise ValueError(f"Missing weather forecast data for {target_date}.")
#         X_scaled_df, _, _, _ = feature_engineer_and_scale(
#             input_df_raw, model_scenario, target_date, x_scaler=x_scaler, y_scaler=y_scaler, change_in_load=predicts_change, apply_scaling=True
#         )
#         X_scaled_target_day = X_scaled_df
#         if X_scaled_target_day.empty:
#             raise ValueError(f"No input feature vector generated for {model_arch}.")
#         if list(X_scaled_target_day.columns) != feature_columns:
#             print(f"Warning: Feature columns mismatch for {model_arch}. Attempting reorder.")
#             try:
#                 X_scaled_target_day = X_scaled_target_day[feature_columns]
#             except KeyError as ke:
#                 missing = set(feature_columns) - set(X_scaled_target_day.columns)
#                 extra = set(X_scaled_target_day.columns) - set(feature_columns)
#                 raise ValueError(f"Feature mismatch for {model_arch}. Missing: {missing}. Extra: {extra}") from ke
#         print(f"Generating scaled predictions for {model_arch}...")
#         is_lstm = KERAS_AVAILABLE and isinstance(model, tf.keras.Model) and any(isinstance(layer, tf.keras.layers.LSTM) for layer in model.layers)
#         if is_lstm:
#             X_input_final = X_scaled_target_day.values.reshape((1, 1, X_scaled_target_day.shape[1]))
#         else:
#             X_input_final = X_scaled_target_day.values
#         y_pred_scaled = model.predict(X_input_final)
#         print(f"Inverse transforming predictions for {model_arch}...")
#         if y_pred_scaled.shape[1] != y_scaler.n_features_in_:
#             raise ValueError(f"Prediction shape mismatch for inverse transform ({model_arch}).")
#         y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
#         final_prediction_original = y_pred_original
#         if predicts_change:
#             print(f"Converting change_in_load prediction for {model_arch}...")
#             X_original_target_day, _, _, _ = feature_engineer_and_scale(
#                 input_df_raw, model_scenario, target_date, x_scaler=x_scaler, y_scaler=y_scaler, change_in_load=predicts_change, apply_scaling=False
#             )
#             if X_original_target_day.empty:
#                 raise ValueError(f"Could not retrieve original X data for change conversion ({model_arch}).")
#             final_prediction_original = convert_change_in_load_to_base_load(X_original_target_day, y_pred_original)
#         # --- End of Base Model block ---

#     # --- Case 3: Unknown Artifact Type ---
#     else:
#         raise TypeError(f"Loaded artifact for {model_arch} (Version: {actual_version_used}) has unknown structure.")

#     # --- Store result in cache and return ---
#     if final_prediction_original is None:
#         raise ValueError(f"Failed to generate final prediction for {cache_key}")

#     # Return prediction, target columns, and the specific model ID used
#     result = (final_prediction_original, target_columns_final, actual_model_id)
#     _prediction_cache[cache_key] = result
#     print(f"Prediction computed and cached for: {cache_key}")
#     return result


# Forecaster_Utils.py or main script

# Global cache for predictions within a single run_forecast execution
_prediction_cache = {}


def get_prediction(feeder_id, target_date, architecture, scenario, version=None):
    """
    Gets the final (original scale) prediction for a given model, handling recursion for RLS
    and fetching actuals needed for RLS adaptation during prediction.
    Applies scaling before RLS combination.
    Uses a cache to avoid re-computing predictions for the same model/date/version.
    Requires input models for RLS stages to have the SAME version string.
    """
    global _prediction_cache
    cache_key = (feeder_id, target_date, architecture, scenario, version)
    if cache_key in _prediction_cache:
        print(f"Cache HIT for: {cache_key}")
        return _prediction_cache[cache_key]
    else:
        print(f"Cache MISS for: {cache_key}. Computing prediction...")

    # 1. Select Model Metadata
    model_metadata = select_model_for_forecast(feeder_id, architecture, scenario, version=version)
    if not model_metadata:
        if version:
            raise ValueError(f"Could not find model metadata for specific version {version} and criteria {cache_key}")
        else:
            raise ValueError(f"Could not find any model metadata for {cache_key}")

    actual_model_id = model_metadata["model_id"]
    artifact_path_info = model_metadata["model_artifact_path"]
    model_arch = model_metadata["model_architecture_type"]
    model_scenario = model_metadata["scenario_type"]
    actual_version_used = model_metadata["model_version"]
    feature_config = model_metadata.get("feature_engineering_config", {})
    predicts_change = feature_config.get("target") == "Net_Load_Change" or "Change_in_Load" in model_arch

    # 2. Load Artifact(s)
    loaded_artifact = load_artifact_from_storage(artifact_path_info)

    # 3. Check Artifact Type and Predict
    final_prediction_original = None
    target_columns_final = loaded_artifact.get("target_columns")
    y_scaler_to_use = None  # Store the relevant y_scaler

    # --- Case 1: RLS Filters Artifact ---
    if "rls_filters" in loaded_artifact and loaded_artifact.get("rls_filters") is not None:
        print(f"Processing RLS artifact for {model_arch} (Version: {actual_version_used})...")
        rls_filters = loaded_artifact["rls_filters"]
        target_columns_final = loaded_artifact.get("target_columns")
        if not target_columns_final:
            raise ValueError("RLS artifact missing 'target_columns'.")

        # --- Fetch Actual Data Needed for RLS Adaptation ---
        print(f"Fetching actual base load data for RLS adaptation (Target Date: {target_date})...")
        start_fetch_dt_actuals = datetime.combine(target_date - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        end_fetch_dt_actuals = datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        actuals_df_raw = fetch_data(feeder_id, start_fetch_dt_actuals.isoformat(), end_fetch_dt_actuals.isoformat())
        if actuals_df_raw.empty:
            raise ValueError(f"Insufficient data fetched for actuals on {target_date}.")

        # --- Get Predictions for Input Models (Original Scale) ---
        input_models = feature_config.get("input_models")
        if not input_models or len(input_models) != 2:
            raise ValueError(f"Invalid 'input_models' in feature_config for {model_arch}")
        print(f"RLS requires inputs from: {input_models}")
        input_pred_1_result = get_prediction(feeder_id, target_date, input_models[0], scenario, actual_version_used)
        input_pred_2_result = get_prediction(feeder_id, target_date, input_models[1], scenario, actual_version_used)
        if input_pred_1_result is None or input_pred_2_result is None:
            raise RuntimeError(f"Failed to get predictions for input models ({model_arch} V:{actual_version_used})")
        input_pred_1_orig = input_pred_1_result[0]
        input_pred_2_orig = input_pred_2_result[0]

        # --- Need a y_scaler to process actuals and scale inputs ---
        # Let's try to get it from one of the base model artifacts loaded recursively (it should be cached)
        # We assume base models (like ANN_Baseload) store the correct y_scaler
        base_model_cache_key = (feeder_id, target_date, input_models[0], scenario, actual_version_used)  # Assuming first input is base
        if base_model_cache_key in _prediction_cache:
            # Need to load the artifact again briefly to get the scaler if not already stored separately
            # This is inefficient - ideally cache the loaded artifact components
            print(f"Retrieving y_scaler from cached base model run: {input_models[0]}")
            base_metadata_temp = select_model_for_forecast(feeder_id, input_models[0], scenario, version=actual_version_used)
            if base_metadata_temp:
                loaded_base_artifact_temp = load_artifact_from_storage(base_metadata_temp["model_artifact_path"])
                y_scaler_to_use = loaded_base_artifact_temp.get("y_scaler")
        if y_scaler_to_use is None:
            # Fallback: Load base model artifact directly if not cached (shouldn't happen often with cache)
            print(f"Fallback: Loading base model {input_models[0]} artifact to get y_scaler...")
            base_metadata_temp = select_model_for_forecast(feeder_id, input_models[0], scenario, version=actual_version_used)
            if base_metadata_temp:
                loaded_base_artifact_temp = load_artifact_from_storage(base_metadata_temp["model_artifact_path"])
                y_scaler_to_use = loaded_base_artifact_temp.get("y_scaler")

        if y_scaler_to_use is None:
            raise ValueError(f"Could not obtain y_scaler needed for RLS processing for {model_arch}")

        # --- Prepare Actuals (Original Scale) ---
        # Use feature_engineer to get correctly shaped original target data
        _, y_actual_original_df, _, _ = feature_engineer_and_scale(
            actuals_df_raw,
            model_scenario,
            target_date,
            x_scaler=None,
            y_scaler=y_scaler_to_use,  # Pass scaler for internal consistency checks if any
            change_in_load=False,
            apply_scaling=False,
        )
        y_actual_original_target_day = y_actual_original_df[y_actual_original_df.index.date == target_date]
        if y_actual_original_target_day.empty:
            raise ValueError(f"Could not extract actual target values for {target_date}.")
        if list(y_actual_original_target_day.columns) != target_columns_final:
            print("Warning: Actuals columns mismatch RLS target columns. Reordering.")
            y_actual_original_target_day = y_actual_original_target_day[target_columns_final]
        actuals_np_orig = y_actual_original_target_day.values
        # --- End Prepare Actuals ---

        # Check shape consistency before combining
        if input_pred_1_orig.shape != actuals_np_orig.shape or input_pred_2_orig.shape != actuals_np_orig.shape:
            raise ValueError(
                f"Shape mismatch before RLS combination: Actuals={actuals_np_orig.shape}, Pred1={input_pred_1_orig.shape}, Pred2={input_pred_2_orig.shape}"
            )

        # Combine predictions using RLS (function now handles scaling internally)
        print(f"Combining predictions for {model_arch} using RLS filters (with internal scaling/adaptation)...")
        final_prediction_original = predict_with_padasip_rls(
            rls_filters,
            y_scaler_to_use,  # Pass the scaler
            actuals_np_orig,  # Pass original actuals
            input_pred_1_orig,  # Pass original predictions
            input_pred_2_orig,
        )

    # --- Case 2: Base Model Artifact (LGBM, Keras) ---
    elif "model" in loaded_artifact and loaded_artifact.get("model") is not None:
        # --- This block remains the same - it produces original scale predictions ---
        print(f"Processing base model artifact for {model_arch} (Version: {actual_version_used})...")
        model = loaded_artifact["model"]
        x_scaler = loaded_artifact["x_scaler"]
        y_scaler = loaded_artifact["y_scaler"]
        feature_columns = loaded_artifact["feature_columns"]
        target_columns_final = loaded_artifact.get("target_columns")
        if model is None or x_scaler is None or y_scaler is None or feature_columns is None or target_columns_final is None:
            raise ValueError(f"Loaded artifact for base model {model_arch} (Version: {actual_version_used}) is missing required components.")
        start_fetch_dt = datetime.combine(target_date - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        end_fetch_dt = datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        input_df_raw = fetch_data(feeder_id, start_fetch_dt.isoformat(), end_fetch_dt.isoformat())
        if input_df_raw.empty:
            raise ValueError(f"Insufficient data fetched for {model_arch} prediction.")
        target_day_data = input_df_raw[input_df_raw.index.date == target_date]
        required_forecast_cols = ["Temperature_Forecast", "Shortwave_Radiation_Forecast", "Cloud_Cover_Forecast"]
        if target_day_data.empty or target_day_data[required_forecast_cols].isnull().values.any():
            raise ValueError(f"Missing weather forecast data for {target_date}.")
        X_scaled_df, _, _, _ = feature_engineer_and_scale(
            input_df_raw, model_scenario, target_date, x_scaler=x_scaler, y_scaler=y_scaler, change_in_load=predicts_change, apply_scaling=True
        )
        X_scaled_target_day = X_scaled_df
        if X_scaled_target_day.empty:
            raise ValueError(f"No input feature vector generated for {model_arch}.")
        if list(X_scaled_target_day.columns) != feature_columns:
            print(f"Warning: Feature columns mismatch for {model_arch}. Attempting reorder.")
            try:
                X_scaled_target_day = X_scaled_target_day[feature_columns]
            except KeyError as ke:
                missing = set(feature_columns) - set(X_scaled_target_day.columns)
                extra = set(X_scaled_target_day.columns) - set(feature_columns)
                raise ValueError(f"Feature mismatch for {model_arch}. Missing: {missing}. Extra: {extra}") from ke
        print(f"Generating scaled predictions for {model_arch}...")
        is_lstm = KERAS_AVAILABLE and isinstance(model, tf.keras.Model) and any(isinstance(layer, tf.keras.layers.LSTM) for layer in model.layers)
        if is_lstm:
            X_input_final = X_scaled_target_day.values.reshape((1, 1, X_scaled_target_day.shape[1]))
        else:
            X_input_final = X_scaled_target_day.values
        y_pred_scaled = model.predict(X_input_final)
        print(f"Inverse transforming predictions for {model_arch}...")
        if y_pred_scaled.shape[1] != y_scaler.n_features_in_:
            raise ValueError(f"Prediction shape mismatch for inverse transform ({model_arch}).")
        y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
        final_prediction_original = y_pred_original
        if predicts_change:
            print(f"Converting change_in_load prediction for {model_arch}...")
            X_original_target_day, _, _, _ = feature_engineer_and_scale(
                input_df_raw, model_scenario, target_date, x_scaler=x_scaler, y_scaler=y_scaler, change_in_load=predicts_change, apply_scaling=False
            )
            if X_original_target_day.empty:
                raise ValueError(f"Could not retrieve original X data for change conversion ({model_arch}).")
            final_prediction_original = convert_change_in_load_to_base_load(X_original_target_day, y_pred_original)
        # Store the y_scaler used by this base model for potential use by RLS later
        y_scaler_to_use = y_scaler
        # --- End of Base Model block ---

    # --- Case 3: Unknown Artifact Type ---
    else:
        raise TypeError(f"Loaded artifact for {model_arch} (Version: {actual_version_used}) has unknown structure.")

    # --- Store result in cache and return ---
    if final_prediction_original is None:
        raise ValueError(f"Failed to generate final prediction for {cache_key}")

    # Return prediction, target columns, model ID, and the y_scaler used (for RLS case)
    result = (final_prediction_original, target_columns_final, actual_model_id, y_scaler_to_use)
    _prediction_cache[cache_key] = result
    print(f"Prediction computed and cached for: {cache_key}")
    return result
