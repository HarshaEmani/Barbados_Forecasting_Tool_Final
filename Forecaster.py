import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import json
import pickle
import os
import sys
import argparse
from sklearn.preprocessing import MinMaxScaler
from supabase import create_client, Client
import traceback
from dotenv import load_dotenv, find_dotenv

# Ensure TensorFlow/Keras is installed
try:
    import tensorflow as tf
    from keras.models import load_model

    KERAS_AVAILABLE = True
except ImportError:
    print("TensorFlow/Keras not found. Keras models cannot be loaded.")
    KERAS_AVAILABLE = False
    load_model = None

# Ensure padasip is installed (optional)
try:
    from padasip.filters import FilterRLS
    from Forecaster_Utils import run_forecast

    PADASIP_AVAILABLE = True
except ImportError:
    print("padasip not found. RLS filters cannot be loaded/used.")
    PADASIP_AVAILABLE = False
    FilterRLS = object

load_dotenv()  # Load environment variables from .env file

print(find_dotenv())  # Check if .env file is found


# --- Configuration & Constants --- (Should match training script)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SECRET_KEY")
if not SUPABASE_URL or not SUPABASE_KEY or "YOUR_SUPABASE_URL" in SUPABASE_URL:
    print("Error: SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
    # sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
DATA_SCHEMA = "data"
ML_SCHEMA = "ml"
STORAGE_BUCKET = "models"
DAY_HOURS = list(range(6, 20 + 1))
NIGHT_HOURS = list(range(0, 6)) + list(range(21, 24))
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
TEMP_DIR = os.path.join(script_dir, "tmp")


# --- Reusable Functions ---
# (fetch_data, prepare_daily_vectors, feature_engineer_and_scale, convert_change_in_load_to_base_load)

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


# def prepare_daily_vectors(df, feature_cols, target_col_list, scenario_hours):
#     """Pivots hourly data into daily vectors (one row per day)."""
#     print("Pivoting data into daily vectors...")
#     df_copy = df.copy()
#     df_copy["date"] = df_copy.index.date
#     df_copy["hour"] = df_copy.index.hour
#     if scenario_hours:
#         df_copy = df_copy[df_copy["hour"].isin(scenario_hours)]
#     pivoted_X = df_copy.pivot_table(index="date", columns="hour", values=feature_cols)
#     pivoted_X.columns = [f"{col[0]}_Hour_{col[1]}" for col in pivoted_X.columns]
#     try:
#         pivoted_y = df_copy.pivot_table(index="date", columns="hour", values=target_col_list)
#         if len(target_col_list) == 1:
#             if isinstance(pivoted_y, pd.Series):
#                 pivoted_y = pivoted_y.to_frame()
#             pivoted_y.columns = [f"{target_col_list[0]}_Hour_{col[1]}" for col in pivoted_y.columns]
#         else:
#             pivoted_y.columns = [f"{col[0]}_Hour_{col[1]}" for col in pivoted_y.columns]
#     except Exception as e:
#         print(f"Warning: Could not pivot target columns {target_col_list} (may not exist in forecast input): {e}")
#         pivoted_y = pd.DataFrame(index=pivoted_X.index)
#     expected_hours = scenario_hours if scenario_hours else list(range(24))
#     ordered_X_columns = [f"{feat}_Hour_{hr}" for feat in feature_cols for hr in expected_hours]
#     pivoted_X = pivoted_X.reindex(columns=ordered_X_columns)
#     pivoted_X.index = pd.to_datetime(pivoted_X.index)
#     pivoted_X["DOW"] = pivoted_X.index.dayofweek
#     pivoted_X["Is_Holiday"] = 0
#     pivoted_X = pd.get_dummies(pivoted_X, columns=["DOW"], prefix="DOW", dtype="int")

#     # --- Ensure all DOW columns exist ---
#     expected_dow_cols = [f"DOW_{i}" for i in range(7)]
#     for col in expected_dow_cols:
#         if col not in pivoted_X.columns:
#             print(f"Adding missing column: {col}")
#             pivoted_X[col] = 0  # Add missing DOW columns and set to 0
#     # Ensure correct dtype after adding potentially missing columns
#     pivoted_X[expected_dow_cols] = pivoted_X[expected_dow_cols].astype(int)

#     pivoted_X["Is_Weekend"] = pivoted_X["DOW_5"] + pivoted_X["DOW_6"]

#     DOW_ordered_columns = [f"DOW_{i}" for i in range(7)]
#     pivoted_X = pivoted_X.reindex(columns=ordered_X_columns + DOW_ordered_columns + ["Is_Holiday", "Is_Weekend"])

#     print(pivoted_X.columns)

#     print(f"Reshaped data: X shape {pivoted_X.shape}")
#     return pivoted_X, pivoted_y


# def feature_engineer_and_scale(df, scenario, target_date, x_scaler=None, y_scaler=None, change_in_load=False, apply_scaling=True):
#     """Prepares features, reshapes data, applies MinMaxScaler to X and y."""
#     print(f"Starting feature engineering for scenario: {scenario}...")
#     df_processed = df.copy()
#     df_processed["Net_Load_Change"] = df_processed["Net_Load_Demand"].diff(24).fillna(0)
#     df_processed["Prev_Day_Net_Load_Demand"] = df_processed["Net_Load_Demand"].shift(24)
#     df_processed["Prev_Day_Temperature_Historic"] = df_processed["Temperature_Historic"].shift(24)
#     df_processed["Prev_Day_Shortwave_Radiation_Historic"] = df_processed["Shortwave_Radiation_Historic"].shift(24)
#     df_processed["Prev_Day_Cloud_Cover_Historic"] = df_processed["Cloud_Cover_Historic"].shift(24)
#     # df_processed.rename(
#     #     columns={
#     #         "temperature_2m_forecast": "Temperature_Forecast",
#     #         "shortwave_radiation_forecast": "Shortwave_Radiation_Forecast",
#     #         "cloud_cover_forecast": "Cloud_Cover_Forecast",
#     #     },
#     #     inplace=True,
#     # )
#     feature_cols = [
#         "Prev_Day_Net_Load_Demand",
#         "Prev_Day_Temperature_Historic",
#         "Temperature_Forecast",
#         "Prev_Day_Shortwave_Radiation_Historic",
#         "Shortwave_Radiation_Forecast",
#         "Prev_Day_Cloud_Cover_Historic",
#         "Cloud_Cover_Forecast",
#     ]
#     target_col = ["Net_Load_Change"] if change_in_load else ["Net_Load_Demand"]
#     scenario_hours = DAY_HOURS if scenario == "Day" else NIGHT_HOURS if scenario == "Night" else None

#     df_processed = df_processed.dropna()

#     X, _ = prepare_daily_vectors(df_processed, feature_cols, target_col, scenario_hours)

#     # --- Filter for Target Date ---
#     # Do this *after* prepare_daily_vectors has created DOW columns
#     X = X[X.index.date == target_date]

#     if X.empty:
#         print(f"Warning: No data row found for target date {target_date} after reshaping.")
#         # Return empty dataframe matching expected structure if possible, else None
#         # This depends on whether feature_columns are known at this point
#         return pd.DataFrame(), None, None, None

#     print("+++++++++++++++++++++++++")
#     print(list(X.columns))
#     print("+++++++++++++++++++++++++")

#     if X.empty:
#         print(f"Warning: No data row found for target date after reshaping.")
#         return X, None, None, None
#     if X.isnull().values.any():
#         print("ERROR: NaNs detected in the final input vector for forecasting!")
#         print(X[X.isnull().any(axis=1)])
#         raise ValueError("NaNs found in input features for forecast day. Check data fetching and feature engineering.")
#     if not apply_scaling:
#         print("Scaling is disabled.")
#         return X, None, None, None
#     if x_scaler is None:
#         raise ValueError("x_scaler must be provided for forecasting mode.")
#     print("Transforming input features (X) using provided scaler...")
#     if not hasattr(x_scaler, "transform"):
#         raise ValueError("Provided x_scaler object must have a 'transform' method.")
#     try:
#         if hasattr(x_scaler, "feature_names_in_") and list(X.columns) != list(x_scaler.feature_names_in_):
#             print("Warning: Feature mismatch detected. Attempting reorder...")
#             X = X[x_scaler.feature_names_in_]
#         elif hasattr(x_scaler, "n_features_in_") and X.shape[1] != x_scaler.n_features_in_:
#             raise ValueError(f"Input feature count mismatch: data has {X.shape[1]}, X scaler expects {x_scaler.n_features_in_}")
#         X_scaled = x_scaler.transform(X)
#         print("Input features transformed.")
#         X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
#         return X_scaled_df, None, None, None
#     except Exception as e:
#         print(f"Error applying scaler transform: {e}")
#         traceback.print_exc()
#         raise

# def convert_change_in_load_to_base_load(X_original, y_pred_change_original):
#     """Converts predicted change_in_load back to base_load."""
#     X_original_np = X_original.values if isinstance(X_original, pd.DataFrame) else X_original
#     y_pred_change_np = y_pred_change_original.values if isinstance(y_pred_change_original, pd.DataFrame) else y_pred_change_original
#     prev_day_cols = [col for col in X_original.columns if col.startswith("Prev_Day_Net_Load_Demand_Hour_")]
#     if len(prev_day_cols) == 0:
#         raise ValueError("Could not find 'Prev_Day_Net_Load_Demand_Hour_' columns in X_original.")
#     if len(prev_day_cols) != y_pred_change_np.shape[1]:
#         raise ValueError(
#             f"Mismatch between number of previous day load columns ({len(prev_day_cols)}) and prediction columns ({y_pred_change_np.shape[1]})"
#         )
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

#         # Add filters based on provided arguments
#         if architecture:
#             query = query.eq("model_architecture_type", architecture)
#         if scenario:
#             query = query.eq("scenario_type", scenario)
#         # --- Use exact version match ---
#         if version:
#             print(f"Filtering for EXACT version: {version}")
#             query = query.eq("model_version", version)
#         # --- End exact version match ---

#         # If version is specified, we expect only one match.
#         # If version is NOT specified, get the latest matching the other criteria.
#         if not version:
#             query = query.order("training_timestamp", desc=True).limit(1)

#         response = query.execute()

#         # Handle potential multiple matches if version wasn't specific enough (shouldn't happen with exact match)
#         if response.data and len(response.data) > 1 and version:
#             print(f"Warning: Found multiple models ({len(response.data)}) matching exact version '{version}'. Using the first one found.")

#         if response.data:
#             model_metadata = response.data[0]
#             print(
#                 f"Selected Model ID: {model_metadata.get('model_id')}, Version: {model_metadata.get('model_version')}, Path Info: {model_metadata.get('model_artifact_path')}"
#             )
#             # Parse JSON fields
#             for key in ["model_hyperparameters", "feature_engineering_config", "validation_metrics"]:
#                 if model_metadata.get(key) and isinstance(model_metadata[key], str):
#                     try:
#                         model_metadata[key] = json.loads(model_metadata[key])
#                     except json.JSONDecodeError:
#                         print(f"Warning: Could not parse JSON for {key}")
#             # Special parsing for model_artifact_path if it's JSON (Keras case)
#             if isinstance(model_metadata.get("model_artifact_path"), str) and model_metadata["model_artifact_path"].startswith("{"):
#                 try:
#                     model_metadata["model_artifact_path"] = json.loads(model_metadata["model_artifact_path"])
#                 except json.JSONDecodeError:
#                     print(f"Warning: Could not parse JSON for model_artifact_path")
#             return model_metadata
#         else:
#             print(f"Error: No matching model found for criteria: Feeder={feeder_id}, Arch={architecture}, Scenario={scenario}, Version={version}")
#             return None
#     except Exception as e:
#         print(f"Error selecting model metadata: {e}")
#         raise


# def store_forecasts(model_id, feeder_id, forecast_run_timestamp, target_date, predictions_original, target_columns):
#     """Formats and inserts forecast results into ml.Forecasts."""
#     print(f"Storing forecasts for Feeder {feeder_id}, Target Date: {target_date}, Model ID: {model_id}")
#     if predictions_original is None or predictions_original.size == 0:
#         print("Warning: No predictions provided to store.")
#         return
#     try:
#         actual_hours = sorted([int(col.split("_Hour_")[-1]) for col in target_columns])
#         if len(actual_hours) != predictions_original.shape[1]:
#             raise ValueError("Mismatch between target columns and prediction shape.")
#         target_timestamps = [datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=h) for h in actual_hours]
#         forecast_values = predictions_original.flatten()
#         if len(forecast_values) != len(target_timestamps):
#             raise ValueError(f"Mismatch between number of forecast values ({len(forecast_values)}) and target timestamps ({len(target_timestamps)}).")
#         records_to_insert = [
#             {
#                 "model_id": model_id,
#                 "feeder_id": feeder_id,
#                 "forecast_run_timestamp": forecast_run_timestamp,
#                 "target_timestamp": ts.isoformat(),
#                 "forecast_value": float(forecast_values[i]),
#                 "actual_value": None,
#             }
#             for i, ts in enumerate(target_timestamps)
#         ]
#         supabase.postgrest.schema(ML_SCHEMA)
#         response = supabase.table("forecasts").insert(records_to_insert).execute()
#         if hasattr(response, "data") and response.data:
#             print(f"Successfully inserted {len(response.data)} forecast records.")
#         elif hasattr(response, "error") and response.error:
#             print(f"Error inserting forecasts: {response.error}")
#             raise Exception(f"Failed to insert forecasts: {response.error}")
#         elif not hasattr(response, "error") and not hasattr(response, "data"):
#             print(f"Forecasts inserted (assumed based on response, count unknown).")
#         else:
#             print(f"Unknown error inserting forecasts. Response: {response}")
#             raise Exception("Unknown error inserting forecasts.")
#     except Exception as e:
#         print(f"Error formatting or storing forecasts: {e}")
#         traceback.print_exc()
#         raise


# def load_artifact_from_storage(artifact_path_info):
#     """Downloads artifact(s) from Supabase Storage and loads them."""
#     print(f"Loading artifact(s) based on path info: {artifact_path_info}")
#     os.makedirs(TEMP_DIR, exist_ok=True)
#     keras_model_path, scalers_pkl_path, single_pkl_path = None, None, None
#     local_keras_temp_path, local_scalers_temp_path, local_single_pkl_temp_path = None, None, None
#     if isinstance(artifact_path_info, dict):
#         keras_model_path = artifact_path_info.get("keras_model")
#         scalers_pkl_path = artifact_path_info.get("scalers_pkl")
#         if not keras_model_path or not scalers_pkl_path:
#             raise ValueError("Invalid artifact path info dictionary.")
#         print(f"Detected separate Keras model ({keras_model_path}) and scalers ({scalers_pkl_path}).")
#         local_keras_temp_path = os.path.join(TEMP_DIR, os.path.basename(keras_model_path))
#         local_scalers_temp_path = os.path.join(TEMP_DIR, os.path.basename(scalers_pkl_path))
#     elif isinstance(artifact_path_info, str) and artifact_path_info.endswith(".pkl"):
#         single_pkl_path = artifact_path_info
#         print(f"Detected single pickle artifact path: {single_pkl_path}")
#         local_single_pkl_temp_path = os.path.join(TEMP_DIR, os.path.basename(single_pkl_path))
#     else:
#         raise ValueError(f"Unsupported artifact path info format: {artifact_path_info}")
#     loaded_model, loaded_scalers_info = None, {}
#     try:
#         if keras_model_path:
#             if not KERAS_AVAILABLE:
#                 raise RuntimeError("Keras artifact specified, but TensorFlow/Keras is not installed.")
#             print(f"Downloading Keras model to: {local_keras_temp_path}")
#             with open(local_keras_temp_path, "wb+") as f:
#                 res = supabase.storage.from_(STORAGE_BUCKET).download(keras_model_path)
#                 f.write(res)
#             print("Keras model downloaded. Loading...")
#             loaded_model = load_model(local_keras_temp_path)
#             print("Keras model loaded.")
#         if scalers_pkl_path:
#             print(f"Downloading scalers pickle to: {local_scalers_temp_path}")
#             with open(local_scalers_temp_path, "wb+") as f:
#                 res = supabase.storage.from_(STORAGE_BUCKET).download(scalers_pkl_path)
#                 f.write(res)
#             print("Scalers pickle downloaded. Loading...")
#             with open(local_scalers_temp_path, "rb") as f:
#                 loaded_scalers_info = pickle.load(f)
#                 print("Scalers pickle loaded.")
#             if not isinstance(loaded_scalers_info, dict) or not all(
#                 k in loaded_scalers_info for k in ["x_scaler", "y_scaler", "feature_columns", "target_columns"]
#             ):
#                 raise TypeError("Loaded scalers pickle file does not contain expected keys.")
#         if single_pkl_path:
#             print(f"Downloading single pickle artifact to: {local_single_pkl_temp_path}")
#             with open(local_single_pkl_temp_path, "wb+") as f:
#                 res = supabase.storage.from_(STORAGE_BUCKET).download(single_pkl_path)
#                 f.write(res)
#             print("Single pickle downloaded. Loading...")
#             with open(local_single_pkl_temp_path, "rb") as f:
#                 loaded_object = pickle.load(f)
#                 print("Single pickle loaded.")
#             if isinstance(loaded_object, dict) and "model" in loaded_object:
#                 loaded_model = loaded_object["model"]
#                 loaded_scalers_info = loaded_object
#             elif PADASIP_AVAILABLE and isinstance(loaded_object.get("rls_filters"), list):
#                 print("Detected RLS filters artifact.")
#                 return {"rls_filters": loaded_object["rls_filters"], "target_columns": loaded_object.get("target_columns")}
#             else:
#                 raise TypeError("Loaded single pickle file has unexpected structure.")
#         if loaded_model is None and "rls_filters" not in loaded_scalers_info:
#             raise ValueError("Failed to load the primary model or RLS filters.")
#         return {
#             "model": loaded_model,
#             "x_scaler": loaded_scalers_info.get("x_scaler"),
#             "y_scaler": loaded_scalers_info.get("y_scaler"),
#             "feature_columns": loaded_scalers_info.get("feature_columns"),
#             "target_columns": loaded_scalers_info.get("target_columns"),
#             "rls_filters": loaded_scalers_info.get("rls_filters"),
#         }  # Include rls_filters if present
#     except Exception as e:
#         print(f"Error loading artifact(s): {e}")
#         traceback.print_exc()
#         raise
#     finally:
#         for temp_path in [local_keras_temp_path, local_scalers_temp_path, local_single_pkl_temp_path]:
#             if temp_path and os.path.exists(temp_path):
#                 try:
#                     os.remove(temp_path)
#                     print(f"Cleaned up temporary file: {temp_path}")
#                 except OSError as rm_err:
#                     print(f"Error removing temporary file {temp_path}: {rm_err}")

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


# def run_forecast(feeder_id, target_date_str, architecture, scenario, version=None):  # Changed version_prefix to version
#     """Orchestrates the forecasting process by calling get_prediction with a specific version."""
#     global _prediction_cache
#     _prediction_cache = {}  # Clear cache for each new run_forecast call

#     print(f"\n--- Starting Forecast Run ---")
#     target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
#     print(f"Feeder: {feeder_id}, Target Date: {target_date}, Arch: {architecture}, Scenario: {scenario}, Version: {version}")
#     forecast_run_timestamp = datetime.now(timezone.utc).isoformat()

#     try:
#         # Get the final prediction using the recursive function
#         # Pass the specific version requested (can be None to get latest)
#         final_prediction_original, target_columns, final_model_id = get_prediction(
#             feeder_id, target_date, architecture, scenario, version=version  # Pass specific version
#         )

#         if final_prediction_original is None:
#             print("ERROR: Failed to obtain final prediction.")
#             return
#         if target_columns is None:
#             print("ERROR: Target column names could not be determined.")
#             return

#         # Store the final result using the ID of the model ultimately selected/used
#         store_forecasts(final_model_id, feeder_id, forecast_run_timestamp, target_date, final_prediction_original, target_columns)

#         print(f"--- Forecast Run Successfully Completed for {architecture}/{scenario} (Version: {version}) ---")

#     except Exception as e:
#         print(f"--- Forecast Run Failed for {architecture}/{scenario} (Version: {version}) ---")
#         print(f"Error: {e}")
#         traceback.print_exc()
#     finally:
#         supabase.postgrest.schema("public")  # Reset schema
#         _prediction_cache = {}  # Clear cache


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


# --- Command-Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Day-Ahead Forecast")
    parser.add_argument("feeder_id", type=int, help="Feeder ID to forecast for")
    parser.add_argument("target_date", type=str, help="Target date for forecast (YYYY-MM-DD)")
    parser.add_argument(
        "--architecture", type=str, required=True, help="Top-level model architecture type to forecast (e.g., Final_RLS_Combined, LSTM_Baseload)"
    )
    parser.add_argument("--scenario", type=str, required=True, help="Scenario type (e.g., 24hr, Day, Night)")
    # Changed back to --version (optional, specific)
    parser.add_argument("--version", type=str, help="Optional: Exact model version string to use.", default=None)

    args = parser.parse_args()
    try:
        datetime.strptime(args.target_date, "%Y-%m-%d")
    except ValueError:
        print("Error: target_date must be in YYYY-MM-DD format.")
        sys.exit(1)

    run_forecast(
        feeder_id=args.feeder_id,
        target_date_str=args.target_date,
        architecture=args.architecture,
        scenario=args.scenario,
        version=args.version,  # Pass specific version
    )
