try:
    # Standard Libraries
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta, timezone
    import json
    import pickle
    import os
    import sys
    import argparse
    import traceback

    # Core ML/Data Libraries
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Supabase & Environment
    from supabase import create_client, Client
    from dotenv import load_dotenv, find_dotenv

    # Specific Model Libraries (Optional based on what's being loaded)
    try:
        import tensorflow as tf
        from keras.models import load_model

        KERAS_AVAILABLE = True
    except ImportError:
        print("TensorFlow/Keras not found. Keras models cannot be loaded.")
        KERAS_AVAILABLE = False
        load_model = None  # Define dummy if not available

    try:
        from padasip.filters import FilterRLS

        PADASIP_AVAILABLE = True
    except ImportError:
        print("padasip not found. RLS filters cannot be loaded/used.")
        PADASIP_AVAILABLE = False
        FilterRLS = object  # Define dummy if not available

    # Your Utility Modules (Ensure these paths are correct)
    # If these are not in modules, paste the function definitions directly
    from Forecaster_Utils import run_forecast
    from DB_Utils import fetch_data, get_all_feeder_ids

    # Assuming run_forecast and its dependencies are defined below or imported
    # from Forecast_Utils import run_forecast, get_prediction, ... # Example if modularized

    np.set_printoptions(suppress=True)

    load_dotenv()
    print("Env file found at location: ", find_dotenv())

except ImportError as e:
    print(f"ERROR: Failed to import necessary libraries: {e}")
    print("Please ensure all required libraries (pandas, numpy, sklearn, supabase, dotenv, tensorflow/keras, padasip, lightgbm) are installed.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during setup: {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Configuration & Constants ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SECRET_KEY")
if not SUPABASE_URL or not SUPABASE_KEY or "YOUR_SUPABASE_URL" in SUPABASE_URL:
    print("Error: SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
DATA_SCHEMA = "data"
ML_SCHEMA = "ml"
METADATA_SCHEMA = "metadata"  # Assuming metadata schema name
STORAGE_BUCKET = "models"
DAY_HOURS = list(range(6, 20 + 1))  # Define these if not imported
NIGHT_HOURS = list(range(0, 6)) + list(range(21, 24))  # Define these if not imported
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
TEMP_DIR = os.path.join(script_dir, "tmp")  # Define TEMP_DIR if not imported

# --- Define Architectures and Scenarios to Forecast ---
# Choose which architectures you want to generate forecasts FOR.
# Typically, you'd run the final ones, maybe a baseline for comparison.
ARCHITECTURES_TO_FORECAST = [
    # "LightGBM_Baseline",
    "LSTM_Change_in_Load",
    # "LSTM_RLS_Combined",
    # Add other architectures (e.g., 'LSTM_Baseload') if needed
]

# SCENARIOS_TO_FORECAST = ["24hr", "Day", "Night"]
SCENARIOS_TO_FORECAST = ["24hr"]


# --- PASTE REQUIRED HELPER FUNCTIONS HERE ---
# You MUST include the definitions for:
# - fetch_data
# - prepare_daily_vectors
# - feature_engineer_and_scale
# - convert_change_in_load_to_base_load
# - select_model_for_forecast
# - load_artifact_from_storage
# - predict_with_padasip_rls
# - get_prediction
# - store_forecasts
# - run_forecast
# --- Make sure they are identical to the versions in your working forecast script ---
# Example placeholder:
# def fetch_data(feeder_id, start_date, end_date):
#     """Fetches combined feeder and weather data from Supabase."""
#     print(f"Fetching data for Feeder {feeder_id} from {start_date} to {end_date}...")
#     try:
#         supabase.postgrest.schema(DATA_SCHEMA)
#         response = (supabase.table(f"Feeder_Weather_Combined_Data").select("*").eq("Feeder_ID", feeder_id).gte("Timestamp", start_date).lt("Timestamp", end_date).order("Timestamp", desc=False).execute())
#         if not response.data: print(f"Warning: No data found for Feeder {feeder_id} in the specified range."); return pd.DataFrame()
#         df = pd.DataFrame(response.data); df["Timestamp"] = pd.to_datetime(df["Timestamp"]); df = df.set_index("Timestamp")
#         print(f"Fetched {len(df)} records."); return df
#     except Exception as e: print(f"Error fetching data: {e}"); raise

# def prepare_daily_vectors(df, feature_cols, target_col_list, scenario_hours):
#     """Pivots hourly data into daily vectors (one row per day)."""
#     print("Pivoting data into daily vectors...")
#     df_copy = df.copy(); df_copy["date"] = df_copy.index.date; df_copy["hour"] = df_copy.index.hour
#     if scenario_hours: df_copy = df_copy[df_copy["hour"].isin(scenario_hours)]
#     pivoted_X = df_copy.pivot_table(index="date", columns="hour", values=feature_cols)
#     pivoted_X.columns = [f"{col[0]}_Hour_{col[1]}" for col in pivoted_X.columns]
#     try:
#         pivoted_y = df_copy.pivot_table(index="date", columns="hour", values=target_col_list)
#         if len(target_col_list) == 1:
#              if isinstance(pivoted_y, pd.Series): pivoted_y = pivoted_y.to_frame()
#              pivoted_y.columns = [f"{target_col_list[0]}_Hour_{col[1]}" for col in pivoted_y.columns]
#         else: pivoted_y.columns = [f"{col[0]}_Hour_{col[1]}" for col in pivoted_y.columns]
#     except Exception as e: print(f"Warning: Could not pivot target columns {target_col_list} (may not exist in forecast input): {e}"); pivoted_y = pd.DataFrame(index=pivoted_X.index)
#     expected_hours = scenario_hours if scenario_hours else list(range(24))
#     ordered_X_columns = [f"{feat}_Hour_{hr}" for feat in feature_cols for hr in expected_hours]
#     pivoted_X = pivoted_X.reindex(columns=ordered_X_columns)
#     pivoted_X.index = pd.to_datetime(pivoted_X.index); pivoted_X["Day_Of_Week"] = pivoted_X.index.dayofweek; pivoted_X["Is_Holiday"] = 0
#     pivoted_X = pd.get_dummies(pivoted_X, columns=["Day_Of_Week"], prefix="DOW", dtype="int")
#     print(f"Reshaped data: X shape {pivoted_X.shape}")
#     return pivoted_X, pivoted_y

# def feature_engineer_and_scale(df, scenario, target_date, x_scaler=None, y_scaler=None, change_in_load=False, apply_scaling=True):
#     """Prepares features, reshapes data, applies MinMaxScaler to X and y. Ensures all DOW columns exist for forecasting."""
#     print(f"Starting feature engineering for scenario: {scenario} for target date: {target_date}...")
#     df_processed = df.copy()
#     df_processed["Net_Load_Change"] = df_processed["Net_Load_Demand"].diff(24).fillna(0)
#     df_processed["Prev_Day_Net_Load_Demand"] = df_processed["Net_Load_Demand"].shift(24)
#     df_processed["Prev_Day_Temperature_Historic"] = df_processed["temperature_2m_historic"].shift(24)
#     df_processed["Prev_Day_Shortwave_Radiation_Historic"] = df_processed["shortwave_radiation_historic"].shift(24)
#     df_processed["Prev_Day_Cloud_Cover_Historic"] = df_processed["cloud_cover_historic"].shift(24)
#     df_processed.rename(columns={'Temperature_Forecast': 'Temperature_Forecast', 'Shortwave_Radiation_Forecast': 'Shortwave_Radiation_Forecast', 'Cloud_Cover_Forecast': 'Cloud_Cover_Forecast'}, inplace=True) # Assuming SQL rename happened
#     feature_cols = ['Prev_Day_Net_Load_Demand', 'Prev_Day_Temperature_Historic', 'Temperature_Forecast', 'Prev_Day_Shortwave_Radiation_Historic', 'Shortwave_Radiation_Forecast', 'Prev_Day_Cloud_Cover_Historic', 'Cloud_Cover_Forecast']
#     target_col = ['Net_Load_Change'] if change_in_load else ['Net_Load_Demand']
#     min_required_date = target_date - timedelta(days=1)
#     df_processed = df_processed[df_processed.index.date >= min_required_date]
#     df_processed = df_processed.dropna(subset=['Prev_Day_Net_Load_Demand'])
#     scenario_hours = DAY_HOURS if scenario == "Day" else NIGHT_HOURS if scenario == "Night" else None
#     X, _ = prepare_daily_vectors(df_processed, feature_cols, target_col, scenario_hours)
#     X = X[X.index.date == target_date]
#     if X.empty: print(f"Warning: No data row found for target date {target_date} after reshaping."); return pd.DataFrame(), None, None, None
#     expected_dow_cols = [f"DOW_{i}" for i in range(7)]
#     for col in expected_dow_cols:
#         if col not in X.columns: print(f"Adding missing column: {col}"); X[col] = 0
#     X[expected_dow_cols] = X[expected_dow_cols].astype(int)
#     if X.isnull().values.any(): print("ERROR: NaNs detected in the final input vector for forecasting!"); print(X[X.isnull().any(axis=1)]); raise ValueError("NaNs found in input features for forecast day.")
#     if not apply_scaling: print("Scaling is disabled."); return X, None, None, None
#     if x_scaler is None: raise ValueError("x_scaler must be provided for forecasting mode.")
#     print("Transforming input features (X) using provided scaler...")
#     if not hasattr(x_scaler, "transform"): raise ValueError("Provided x_scaler object must have a 'transform' method.")
#     try:
#         if hasattr(x_scaler, 'feature_names_in_'): expected_features = list(x_scaler.feature_names_in_)
#         elif hasattr(x_scaler, 'n_features_in_'): print("Warning: Scaler missing feature names."); expected_features = list(X.columns);
#         else: raise ValueError("Scaler object does not provide feature names or count.")
#         for col in expected_features:
#              if col not in X.columns: print(f"Warning: Adding missing expected feature column '{col}' with value 0."); X[col] = 0
#         if list(X.columns) != expected_features:
#              print("Reordering columns to match scaler expectation..."); X = X[expected_features]
#         X_scaled = x_scaler.transform(X); print("Input features transformed.")
#         X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
#         return X_scaled_df, None, None, None
#     except Exception as e: print(f"Error applying scaler transform or checking features: {e}"); traceback.print_exc(); raise

# def convert_change_in_load_to_base_load(X_original, y_pred_change_original):
#     """Converts predicted change_in_load back to base_load."""
#     X_original_np = X_original.values if isinstance(X_original, pd.DataFrame) else X_original
#     y_pred_change_np = y_pred_change_original.values if isinstance(y_pred_change_original, pd.DataFrame) else y_pred_change_original
#     prev_day_cols = [col for col in X_original.columns if col.startswith("Prev_Day_Net_Load_Demand_Hour_")]
#     if len(prev_day_cols) == 0: raise ValueError("Could not find 'Prev_Day_Net_Load_Demand_Hour_' columns in X_original.")
#     if len(prev_day_cols) != y_pred_change_np.shape[1]: raise ValueError(f"Mismatch between number of previous day load columns ({len(prev_day_cols)}) and prediction columns ({y_pred_change_np.shape[1]})")
#     prev_day_indices = [X_original.columns.get_loc(col) for col in prev_day_cols]
#     prev_day_load = X_original_np[:, prev_day_indices]
#     y_pred_base_np = prev_day_load.astype(float) + y_pred_change_np.astype(float)
#     print("Converted change_in_load prediction back to base_load prediction.")
#     return y_pred_base_np

# def select_model_for_forecast(feeder_id, architecture=None, scenario=None, version=None):
#     """Queries ml.Models to find the desired model metadata using an EXACT version match if provided."""
#     print(f"Selecting model for Feeder {feeder_id}, Arch: {architecture}, Scenario: {scenario}, Version: {version}")
#     try:
#         supabase.postgrest.schema(ML_SCHEMA)
#         query = supabase.table("models").select("*").eq("feeder_id", feeder_id)
#         if architecture: query = query.eq("model_architecture_type", architecture)
#         if scenario: query = query.eq("scenario_type", scenario)
#         if version: print(f"Filtering for EXACT version: {version}"); query = query.eq("model_version", version)
#         if not version: query = query.order("training_timestamp", desc=True).limit(1)
#         response = query.execute()
#         if response.data and len(response.data) > 1 and version: print(f"Warning: Found multiple models ({len(response.data)}) matching exact version '{version}'. Using the first one found.")
#         if response.data:
#             model_metadata = response.data[0]
#             print(f"Selected Model ID: {model_metadata.get('model_id')}, Version: {model_metadata.get('model_version')}, Path Info: {model_metadata.get('model_artifact_path')}")
#             for key in ['model_hyperparameters', 'feature_engineering_config', 'validation_metrics']:
#                  if model_metadata.get(key) and isinstance(model_metadata[key], str):
#                      try: model_metadata[key] = json.loads(model_metadata[key])
#                      except json.JSONDecodeError: print(f"Warning: Could not parse JSON for {key}")
#             if isinstance(model_metadata.get('model_artifact_path'), str) and model_metadata['model_artifact_path'].startswith('{'):
#                  try: model_metadata['model_artifact_path'] = json.loads(model_metadata['model_artifact_path'])
#                  except json.JSONDecodeError: print(f"Warning: Could not parse JSON for model_artifact_path")
#             return model_metadata
#         else: print(f"Error: No matching model found for criteria: Feeder={feeder_id}, Arch={architecture}, Scenario={scenario}, Version={version}"); return None
#     except Exception as e: print(f"Error selecting model metadata: {e}"); raise

# def load_artifact_from_storage(artifact_path_info):
#     """Downloads artifact(s) from Supabase Storage and loads them."""
#     print(f"Loading artifact(s) based on path info: {artifact_path_info}")
#     os.makedirs(TEMP_DIR, exist_ok=True)
#     keras_model_path, scalers_pkl_path, single_pkl_path = None, None, None
#     local_keras_temp_path, local_scalers_temp_path, local_single_pkl_temp_path = None, None, None
#     if isinstance(artifact_path_info, dict):
#         keras_model_path = artifact_path_info.get("keras_model"); scalers_pkl_path = artifact_path_info.get("scalers_pkl")
#         if not keras_model_path or not scalers_pkl_path: raise ValueError("Invalid artifact path info dictionary.")
#         print(f"Detected separate Keras model ({keras_model_path}) and scalers ({scalers_pkl_path}).")
#         local_keras_temp_path = os.path.join(TEMP_DIR, os.path.basename(keras_model_path)); local_scalers_temp_path = os.path.join(TEMP_DIR, os.path.basename(scalers_pkl_path))
#     elif isinstance(artifact_path_info, str) and artifact_path_info.endswith(".pkl"):
#         single_pkl_path = artifact_path_info; print(f"Detected single pickle artifact path: {single_pkl_path}"); local_single_pkl_temp_path = os.path.join(TEMP_DIR, os.path.basename(single_pkl_path))
#     else: raise ValueError(f"Unsupported artifact path info format: {artifact_path_info}")
#     loaded_model, loaded_scalers_info = None, {}
#     try:
#         if keras_model_path:
#             if not KERAS_AVAILABLE: raise RuntimeError("Keras artifact specified, but TensorFlow/Keras is not installed.")
#             print(f"Downloading Keras model to: {local_keras_temp_path}");
#             with open(local_keras_temp_path, 'wb+') as f: res = supabase.storage.from_(STORAGE_BUCKET).download(keras_model_path); f.write(res)
#             print("Keras model downloaded. Loading..."); loaded_model = load_model(local_keras_temp_path); print("Keras model loaded.")
#         if scalers_pkl_path:
#             print(f"Downloading scalers pickle to: {local_scalers_temp_path}");
#             with open(local_scalers_temp_path, 'wb+') as f: res = supabase.storage.from_(STORAGE_BUCKET).download(scalers_pkl_path); f.write(res)
#             print("Scalers pickle downloaded. Loading...");
#             with open(local_scalers_temp_path, 'rb') as f: loaded_scalers_info = pickle.load(f); print("Scalers pickle loaded.")
#             if not isinstance(loaded_scalers_info, dict) or not all(k in loaded_scalers_info for k in ['x_scaler', 'y_scaler', 'feature_columns', 'target_columns']): raise TypeError("Loaded scalers pickle file does not contain expected keys.")
#         if single_pkl_path:
#             print(f"Downloading single pickle artifact to: {local_single_pkl_temp_path}");
#             with open(local_single_pkl_temp_path, 'wb+') as f: res = supabase.storage.from_(STORAGE_BUCKET).download(single_pkl_path); f.write(res)
#             print("Single pickle downloaded. Loading...");
#             with open(local_single_pkl_temp_path, 'rb') as f: loaded_object = pickle.load(f); print("Single pickle loaded.")
#             if isinstance(loaded_object, dict) and 'model' in loaded_object: loaded_model = loaded_object['model']; loaded_scalers_info = loaded_object
#             elif PADASIP_AVAILABLE and isinstance(loaded_object.get('rls_filters'), list): print("Detected RLS filters artifact."); return {'rls_filters': loaded_object['rls_filters'], 'target_columns': loaded_object.get('target_columns')}
#             else: raise TypeError("Loaded single pickle file has unexpected structure.")
#         if loaded_model is None and 'rls_filters' not in loaded_scalers_info: raise ValueError("Failed to load the primary model or RLS filters.")
#         return {'model': loaded_model, 'x_scaler': loaded_scalers_info.get('x_scaler'), 'y_scaler': loaded_scalers_info.get('y_scaler'), 'feature_columns': loaded_scalers_info.get('feature_columns'), 'target_columns': loaded_scalers_info.get('target_columns'), 'rls_filters': loaded_scalers_info.get('rls_filters')}
#     except Exception as e: print(f"Error loading artifact(s): {e}"); traceback.print_exc(); raise
#     finally:
#         for temp_path in [local_keras_temp_path, local_scalers_temp_path, local_single_pkl_temp_path]:
#             if temp_path and os.path.exists(temp_path):
#                 try: os.remove(temp_path); print(f"Cleaned up temporary file: {temp_path}")
#                 except OSError as rm_err: print(f"Error removing temporary file {temp_path}: {rm_err}")

# def predict_with_padasip_rls(rls_filters, predictions1, predictions2):
#     """Combines predictions using a list of fitted padasip RLS filters."""
#     if not PADASIP_AVAILABLE: raise RuntimeError("padasip library not found, cannot use RLS filters.")
#     n_samples, n_outputs = predictions1.shape
#     if len(rls_filters) != n_outputs: raise ValueError("Number of RLS filters does not match number of prediction outputs.")
#     combined_predictions = np.zeros_like(predictions1)
#     for t in range(n_samples):
#         for k in range(n_outputs):
#             x_k = np.array([predictions1[t, k], predictions2[t, k]])
#             if np.isnan(x_k).any() or np.isinf(x_k).any(): print(f"Warning: NaN/Inf input for RLS predict at sample {t}, hour {k}. Setting output to NaN."); combined_predictions[t, k] = np.nan; continue
#             try: combined_predictions[t, k] = rls_filters[k].predict(x_k)
#             except Exception as predict_err: print(f"ERROR during RLS predict at sample {t}, hour {k}: {predict_err}"); combined_predictions[t, k] = np.nan
#     return combined_predictions

# _prediction_cache = {}
# def get_prediction(feeder_id, target_date, architecture, scenario, version=None):
#     """Gets the final (original scale) prediction for a given model, handling recursion for RLS."""
#     global _prediction_cache
#     cache_key = (feeder_id, target_date, architecture, scenario, version)
#     if cache_key in _prediction_cache: print(f"Cache HIT for: {cache_key}"); return _prediction_cache[cache_key]
#     else: print(f"Cache MISS for: {cache_key}. Computing prediction...")
#     model_metadata = select_model_for_forecast(feeder_id, architecture, scenario, version=version)
#     if not model_metadata:
#         if version: raise ValueError(f"Could not find model metadata for specific version {version} and criteria {cache_key}")
#         else: raise ValueError(f"Could not find any model metadata for {cache_key}")
#     actual_model_id = model_metadata['model_id']; artifact_path_info = model_metadata['model_artifact_path']; model_arch = model_metadata['model_architecture_type']; model_scenario = model_metadata['scenario_type']; actual_version_used = model_metadata['model_version']; feature_config = model_metadata.get('feature_engineering_config', {}); predicts_change = feature_config.get('target') == 'Net_Load_Change' or 'Change_in_Load' in model_arch
#     loaded_artifact = load_artifact_from_storage(artifact_path_info)
#     final_prediction_original = None; target_columns_final = loaded_artifact.get('target_columns')
#     if 'rls_filters' in loaded_artifact and loaded_artifact.get('rls_filters') is not None:
#         print(f"Processing RLS artifact for {model_arch} (Version: {actual_version_used})..."); rls_filters = loaded_artifact['rls_filters']
#         if not PADASIP_AVAILABLE: raise RuntimeError("padasip library not found, cannot use RLS filters.")
#         input_models = feature_config.get('input_models')
#         if not input_models or len(input_models) != 2: raise ValueError(f"Invalid or missing 'input_models' in feature_config for RLS model {model_arch}")
#         print(f"RLS requires inputs from: {input_models}")
#         input_pred_1_result = get_prediction(feeder_id, target_date, input_models[0], scenario, actual_version_used)
#         input_pred_2_result = get_prediction(feeder_id, target_date, input_models[1], scenario, actual_version_used)
#         if input_pred_1_result is None or input_pred_2_result is None: raise RuntimeError(f"Failed to get predictions for one or both input models for {model_arch} (Version: {actual_version_used})")
#         input_pred_1 = input_pred_1_result[0]; input_pred_2 = input_pred_2_result[0]
#         print(f"Combining predictions for {model_arch} using RLS filters..."); final_prediction_original = predict_with_padasip_rls(rls_filters, input_pred_1, input_pred_2); target_columns_final = loaded_artifact.get('target_columns')
#     elif 'model' in loaded_artifact and loaded_artifact.get('model') is not None:
#         print(f"Processing base model artifact for {model_arch} (Version: {actual_version_used})..."); model = loaded_artifact['model']; x_scaler = loaded_artifact['x_scaler']; y_scaler = loaded_artifact['y_scaler']; feature_columns = loaded_artifact['feature_columns']; target_columns_final = loaded_artifact.get('target_columns')
#         if model is None or x_scaler is None or y_scaler is None or feature_columns is None or target_columns_final is None: raise ValueError(f"Loaded artifact for base model {model_arch} (Version: {actual_version_used}) is missing required components.")
#         start_fetch_dt = datetime.combine(target_date - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc); end_fetch_dt = datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
#         input_df_raw = fetch_data(feeder_id, start_fetch_dt.isoformat(), end_fetch_dt.isoformat())
#         if input_df_raw.empty: raise ValueError(f"Insufficient data fetched for {model_arch} prediction.")
#         target_day_data = input_df_raw[input_df_raw.index.date == target_date]; required_forecast_cols = ['Temperature_Forecast', 'Shortwave_Radiation_Forecast', 'Cloud_Cover_Forecast']
#         if target_day_data.empty or target_day_data[required_forecast_cols].isnull().values.any(): raise ValueError(f"Missing weather forecast data for {target_date}.")
#         X_scaled_df, _, _, _ = feature_engineer_and_scale(input_df_raw, model_scenario, target_date, x_scaler=x_scaler, y_scaler=y_scaler, change_in_load=predicts_change, apply_scaling=True)
#         X_scaled_target_day = X_scaled_df
#         if X_scaled_target_day.empty: raise ValueError(f"No input feature vector generated for {model_arch}.")
#         if list(X_scaled_target_day.columns) != feature_columns:
#              print(f"Warning: Feature columns mismatch for {model_arch}. Attempting reorder.")
#              try: X_scaled_target_day = X_scaled_target_day[feature_columns]
#              except KeyError as ke: missing = set(feature_columns) - set(X_scaled_target_day.columns); extra = set(X_scaled_target_day.columns) - set(feature_columns); raise ValueError(f"Feature mismatch for {model_arch}. Missing: {missing}. Extra: {extra}") from ke
#         print(f"Generating scaled predictions for {model_arch}...")
#         is_lstm = KERAS_AVAILABLE and isinstance(model, tf.keras.Model) and any(isinstance(layer, tf.keras.layers.LSTM) for layer in model.layers)
#         if is_lstm: X_input_final = X_scaled_target_day.values.reshape((1, 1, X_scaled_target_day.shape[1]))
#         else: X_input_final = X_scaled_target_day.values
#         y_pred_scaled = model.predict(X_input_final)
#         print(f"Inverse transforming predictions for {model_arch}...")
#         if y_pred_scaled.shape[1] != y_scaler.n_features_in_: raise ValueError(f"Prediction shape mismatch for inverse transform ({model_arch}).")
#         y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
#         final_prediction_original = y_pred_original
#         if predicts_change:
#             print(f"Converting change_in_load prediction for {model_arch}...")
#             X_original_target_day, _, _, _ = feature_engineer_and_scale(input_df_raw, model_scenario, target_date, x_scaler=x_scaler, y_scaler=y_scaler, change_in_load=predicts_change, apply_scaling=False)
#             if X_original_target_day.empty: raise ValueError(f"Could not retrieve original X data for change conversion ({model_arch}).")
#             final_prediction_original = convert_change_in_load_to_base_load(X_original_target_day, y_pred_original)
#     else: raise TypeError(f"Loaded artifact for {model_arch} (Version: {actual_version_used}) has unknown structure.")
#     if final_prediction_original is None: raise ValueError(f"Failed to generate final prediction for {cache_key}")
#     result = (final_prediction_original, target_columns_final, actual_model_id); _prediction_cache[cache_key] = result; print(f"Prediction computed and cached for: {cache_key}"); return result

# def store_forecasts(model_id, feeder_id, forecast_run_timestamp, target_date, predictions_original, target_columns):
#     """Formats and inserts forecast results into ml.Forecasts."""
#     print(f"Storing forecasts for Feeder {feeder_id}, Target Date: {target_date}, Model ID: {model_id}")
#     if predictions_original is None or predictions_original.size == 0: print("Warning: No predictions provided to store."); return
#     try:
#         actual_hours = sorted([int(col.split('_Hour_')[-1]) for col in target_columns])
#         if len(actual_hours) != predictions_original.shape[1]: raise ValueError("Mismatch between target columns and prediction shape.")
#         target_timestamps = [datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=h) for h in actual_hours]
#         forecast_values = predictions_original.flatten()
#         if len(forecast_values) != len(target_timestamps): raise ValueError(f"Mismatch between number of forecast values ({len(forecast_values)}) and target timestamps ({len(target_timestamps)}).")
#         records_to_insert = [{'model_id': model_id, 'feeder_id': feeder_id, 'forecast_run_timestamp': forecast_run_timestamp, 'target_timestamp': ts.isoformat(), 'forecast_value': float(forecast_values[i]), 'actual_value': None} for i, ts in enumerate(target_timestamps)]
#         supabase.postgrest.schema(ML_SCHEMA)
#         response = supabase.table("forecasts").insert(records_to_insert).execute()
#         if hasattr(response, 'data') and response.data: print(f"Successfully inserted {len(response.data)} forecast records.")
#         elif hasattr(response, 'error') and response.error: print(f"Error inserting forecasts: {response.error}"); raise Exception(f"Failed to insert forecasts: {response.error}")
#         elif not hasattr(response, 'error') and not hasattr(response, 'data'): print(f"Forecasts inserted (assumed based on response, count unknown).")
#         else: print(f"Unknown error inserting forecasts. Response: {response}"); raise Exception("Unknown error inserting forecasts.")
#     except Exception as e: print(f"Error formatting or storing forecasts: {e}"); traceback.print_exc(); raise

# def run_forecast(feeder_id, target_date_str, architecture, scenario, version=None):
#     """Orchestrates the forecasting process by calling get_prediction with a specific version."""
#     global _prediction_cache; _prediction_cache = {}
#     print(f"\n--- Starting Forecast Run ---")
#     target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
#     print(f"Feeder: {feeder_id}, Target Date: {target_date}, Arch: {architecture}, Scenario: {scenario}, Version: {version}")
#     forecast_run_timestamp = datetime.now(timezone.utc).isoformat()
#     try:
#         final_prediction_original, target_columns, final_model_id = get_prediction(feeder_id, target_date, architecture, scenario, version=version)
#         if final_prediction_original is None: print("ERROR: Failed to obtain final prediction."); return
#         if target_columns is None: print("ERROR: Target column names could not be determined."); return
#         store_forecasts(final_model_id, feeder_id, forecast_run_timestamp, target_date, final_prediction_original, target_columns)
#         print(f"--- Forecast Run Successfully Completed for {architecture}/{scenario} (Version: {version}) ---")
#     except Exception as e: print(f"--- Forecast Run Failed for {architecture}/{scenario} (Version: {version}) ---"); print(f"Error: {e}"); traceback.print_exc()
#     finally: supabase.postgrest.schema("public"); _prediction_cache = {}

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Day-Ahead Forecasts for All Feeders")
    parser.add_argument("target_date", type=str, help="Target date for forecast (YYYY-MM-DD)")
    # Version is now optional, if omitted, the latest model will be used
    parser.add_argument("--version", type=str, help="Optional: Exact model version string to use for ALL models.", default=None)

    args = parser.parse_args()

    # Validate target_date format
    try:
        target_date_obj = datetime.strptime(args.target_date, "%Y-%m-%d").date()
    except ValueError:
        print("Error: target_date must be in YYYY-MM-DD format.")
        sys.exit(1)

    print(f"--- Starting Automated Forecasting for Date: {args.target_date} ---")
    if args.version:
        print(f"--- Using Specific Model Version: {args.version} ---")
    else:
        print(f"--- Using LATEST available model version for each combination ---")

    # 1. Get Feeder IDs
    # Assuming get_all_feeder_ids is defined or imported correctly
    try:
        # Need to instantiate the class if get_all_feeder_ids is a method
        # Or call it directly if it's a standalone function
        # Example assuming standalone:
        feeder_ids_to_process = get_all_feeder_ids(supabase)
        # Example assuming it's in Trainer_Utils module:
        # feeder_ids_to_process = Trainer_Utils.get_all_feeder_ids(supabase)
    except NameError:
        print("ERROR: get_all_feeder_ids function not found. Make sure it's defined or imported.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR fetching feeder IDs: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not feeder_ids_to_process:
        print("No Feeder IDs found. Exiting.")
        sys.exit(1)

    # 2. Loop through Feeders, Architectures, and Scenarios
    total_runs = len(feeder_ids_to_process) * len(ARCHITECTURES_TO_FORECAST) * len(SCENARIOS_TO_FORECAST)
    run_counter = 0
    error_counter = 0

    print(f"\nStarting forecast generation loops for {total_runs} total combinations...")

    for feeder_id in feeder_ids_to_process:
        for architecture in ARCHITECTURES_TO_FORECAST:
            for scenario in SCENARIOS_TO_FORECAST:
                run_counter += 1
                print(
                    f"\n--- ({run_counter}/{total_runs}) Processing Forecast: Feeder={feeder_id}, Arch={architecture}, Scenario={scenario}, Version={args.version or 'Latest'} ---"
                )

                try:
                    # Call the main forecast function for this combination
                    # Pass the specific version from args (which might be None)
                    run_forecast(
                        feeder_id=feeder_id,
                        target_date_str=args.target_date,
                        architecture=architecture,
                        scenario=scenario,
                        version=args.version,  # Pass the specific version or None
                    )

                    # sys.exit(1)
                    # Note: run_forecast now handles internal errors and prints messages

                except Exception as e:
                    # Catch errors that might occur *outside* the main try block in run_forecast
                    error_counter += 1
                    print(
                        f"!!! UNHANDLED ERROR during forecast run ({run_counter}/{total_runs}) for Feeder={feeder_id}, Arch={architecture}, Scenario={scenario} !!!"
                    )
                    print(f"Error message: {e}")
                    traceback.print_exc()
                    print(f"--- Skipping to next combination ---")
                    # continue # Continue to the next iteration

    print("\n--- Automated Forecasting Run Finished ---")
    print(f"Total combinations attempted: {run_counter}")
    # Note: Success/failure count is harder to track accurately here as run_forecast handles internal errors.
    # You might need run_forecast to return a status code or check logs for a more precise count.
    print(f"Check logs for success/failure of individual forecast runs.")
