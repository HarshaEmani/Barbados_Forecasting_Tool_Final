try:
    import tensorflow as tf
    from keras.models import Sequential, save_model, load_model
    from keras.layers import Dense, LSTM, Input, Dropout, Layer, Lambda
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
    from supabase import create_client, Client
    import traceback
    import plotly.express as px
    import time
    from dotenv import load_dotenv, find_dotenv

    # from Trainer_Utils import NormalizeLayer

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
# script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
# TEMP_DIR = os.path.join(script_dir, "tmp")
TEMP_DIR = "./../tmp"


class NormalizeLayer(Layer):
    def __init__(self, mean, std, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)
        self.normalize = normalize

    def call(self, inputs):
        if self.normalize:
            return (inputs - self.mean) / (self.std + 1e-8)
        else:
            return (inputs * self.std) + self.mean

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "mean": self.mean.numpy().tolist(),
                "std": self.std.numpy().tolist(),
                "normalize": self.normalize,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        mean = config.pop("mean")
        std = config.pop("std")
        return cls(mean=mean, std=std, **config)


# --- Database Interaction Functions ---
def fetch_data(feeder_id, start_date, end_date):
    """Fetches combined feeder and weather data from Supabase."""
    print(f"Fetching data for Feeder {feeder_id} from {start_date} to {end_date}...")
    end_date_dt = pd.to_datetime(end_date) + timedelta(days=1)  # Include the end date in the range
    end_date_str = end_date_dt.strftime("%Y-%m-%d %H:%M:%S%z")
    try:
        supabase.postgrest.schema(DATA_SCHEMA)
        response = (
            supabase.schema(DATA_SCHEMA)
            .table(f"Feeder_Weather_Combined_Data")
            .select("*")
            .eq("Feeder_ID", feeder_id)
            .gte("Timestamp", start_date)
            .lt("Timestamp", end_date_str)
            .order("Timestamp", desc=False)
            .execute()
        )
        if not response.data:
            print(response.data)
            print(DATA_SCHEMA, feeder_id, start_date, end_date_str)
            print(f"Warning: No data found for Feeder {feeder_id} in the specified range.")
            return pd.DataFrame()
        df = pd.DataFrame(response.data)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.set_index("Timestamp")
        print(f"Fetched {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise


def fetch_feeder_data(feeder_id, start_date, end_date):
    """Fetches combined feeder and weather data from Supabase."""
    print(f"Fetching data for Feeder {feeder_id} from {start_date} to {end_date}...")
    end_date_dt = pd.to_datetime(end_date) + timedelta(days=1)  # Include the end date in the range
    end_date_str = end_date_dt.strftime("%Y-%m-%d %H:%M:%S%z")
    try:
        supabase.postgrest.schema(DATA_SCHEMA)
        response = (
            supabase.schema(DATA_SCHEMA)
            .table(f"Feeder_Weather_Combined_Data")
            .select("Net_Load_Demand, Timestamp")
            .eq("Feeder_ID", feeder_id)
            .gte("Timestamp", start_date)
            .lt("Timestamp", end_date_str)
            .order("Timestamp", desc=False)
            .execute()
        )
        if not response.data:
            print(f"Warning: No data found for Feeder {feeder_id} in the specified range.")
            return pd.DataFrame()
        df = pd.DataFrame(response.data)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.set_index("Timestamp")
        print(f"Fetched {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise


def fetch_input_data_with_options(feeder_id, start_date, end_date, columns=None):
    """Fetches data with optional filtering by columns, model, and scenario."""
    print(f"Fetching data for Feeder {feeder_id} from {start_date} to {end_date}...")
    end_date_dt = pd.to_datetime(end_date) + timedelta(days=1)  # Include the end date in the range
    end_date_str = end_date_dt.strftime("%Y-%m-%d %H:%M:%S%z")
    try:
        supabase.postgrest.schema(DATA_SCHEMA)
        query = (
            supabase.schema(DATA_SCHEMA)
            .table(f"Feeder_Weather_Combined_Data")
            .select("*")
            .eq("Feeder_ID", feeder_id)
            .gte("Timestamp", start_date)
            .lt("Timestamp", end_date_str)
            .order("Timestamp", desc=False)
        )

        if columns:
            query = query.select(columns)

        # if model:
        #     query = query.eq("model_architecture_type", model)

        # if scenario:
        #     query = query.eq("scenario_type", scenario)

        response = query.execute()
        if not response.data:
            print(f"Warning: No data found for Feeder {feeder_id} in the specified range.")
            return pd.DataFrame()
        df = pd.DataFrame(response.data)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.set_index("Timestamp")
        print(f"Fetched {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise


def fetch_forecast_data_with_options(feeder_id, start_date, end_date, columns=None, model=None, scenario=None):
    """Fetches data with optional filtering by columns, model, and scenario."""
    print(f"Fetching data for Feeder {feeder_id} from {start_date} to {end_date}...")

    end_date_dt = pd.to_datetime(end_date) + timedelta(days=1)  # Include the end date in the range
    end_date_str = end_date_dt.strftime("%Y-%m-%d %H:%M:%S%z")
    try:
        supabase.postgrest.schema(ML_SCHEMA)

        model_id_query = (
            supabase.schema(ML_SCHEMA)
            .table("models")
            .select("model_id")
            .eq("feeder_id", feeder_id)
            .eq("model_architecture_type", model)
            .eq("scenario_type", scenario)
            .execute()
        )

        # if model:
        #     model_id_query = model_id_query.eq("model_architecture_type", model)

        # if scenario:
        #     model_id_query = model_id_query.eq("scenario_type", scenario)

        response = model_id_query

        if not response.data:
            print(f"Warning: No model found for Feeder {feeder_id} with specified criteria.")
            return pd.DataFrame()

        model_id = response.data[-1]["model_id"]
        print(f"Model ID: {model_id}")

        query = (
            supabase.schema(ML_SCHEMA)
            .table(f"forecasts")
            .select("*")
            .eq("feeder_id", feeder_id)
            .eq("model_id", model_id)
            .gte("target_timestamp", start_date)
            .lt("target_timestamp", end_date_str)
            .order("target_timestamp", desc=False)
        )

        # if columns:
        #     query = query.select(columns)

        response = query.execute()
        if not response.data:
            print(f"Warning: No data found for Feeder {feeder_id} in the specified range.")
            return pd.DataFrame()
        df = pd.DataFrame(response.data)
        df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])
        df = df.set_index("target_timestamp")
        print(f"Fetched {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise


def get_all_feeder_ids(supabase_client: Client):
    """Fetches all Feeder_ID values from the metadata table."""
    print("Fetching list of Feeder IDs...")
    try:
        supabase_client.postgrest.schema(METADATA_SCHEMA)
        response = supabase_client.schema(METADATA_SCHEMA).table("Feeders_Metadata").select("Feeder_ID").execute()
        if response.data:
            feeder_ids = [item["Feeder_ID"] for item in response.data]
            print(f"Found {len(feeder_ids)} feeders: {feeder_ids}")
            return feeder_ids
        else:
            print("Warning: No feeders found in metadata table.")
            return []
    except Exception as e:
        print(f"Error fetching feeder IDs: {e}")
        traceback.print_exc()
        return []
    finally:
        supabase_client.postgrest.schema("public")  # Reset schema


def log_model_metadata(metadata):
    """Inserts model metadata into the ml.Models table."""
    print(f"Logging metadata for model: {metadata.get('model_artifact_path')}")
    try:
        supabase.postgrest.schema(ML_SCHEMA)
        response = supabase.schema(ML_SCHEMA).table(f"models").insert(metadata).execute()
        if hasattr(response, "data") and response.data:
            print("Metadata logged successfully.")
            return response.data[0]["model_id"]
        elif hasattr(response, "error") and response.error:
            print(f"Error logging metadata: {response.error}")
            raise Exception(f"Failed to log model metadata: {response.error}")
        elif not hasattr(response, "error") and not hasattr(response, "data"):
            print("Metadata logged successfully (assumed based on response).")
            return None
        else:
            print(f"Unknown error logging metadata. Response: {response}")
            raise Exception("Unknown error logging metadata.")
    except Exception as e:
        print(f"Error inserting metadata into {ML_SCHEMA}.Models: {e}")
        raise


# --- save_pickle_artifact (Renamed & Simplified: ONLY handles pickling) ---
def save_pickle_artifact(artifact_object, feeder_id, model_arch, scenario, version_tag):
    """Saves a Python object using pickle to Supabase Storage."""
    file_name = f"{model_arch}_{scenario}_{version_tag}.pkl"  # Use specific tag
    storage_path = f"models/feeder_{feeder_id}/{file_name}"
    local_tmp_dir = TEMP_DIR  # Use the globally defined TEMP_DIR
    local_temp_path = os.path.join(local_tmp_dir, file_name)

    os.makedirs(local_tmp_dir, exist_ok=True)
    print(f"Saving pickled artifact temporarily to: {local_temp_path}")
    print(f"Uploading pickled artifact to Supabase Storage path: {storage_path}...")
    try:
        # 1. Save artifact locally using pickle
        with open(local_temp_path, "wb") as f:
            pickle.dump(artifact_object, f)
        print(f"Artifact pickled locally: {local_temp_path}")

        # 2. Upload the local file to Supabase Storage
        if not os.path.exists(local_temp_path):
            raise FileNotFoundError(f"Temporary file not found after saving: {local_temp_path}")
        with open(local_temp_path, "rb") as f:
            response = supabase.storage.from_(STORAGE_BUCKET).upload(
                path=storage_path, file=f, file_options={"content-type": "application/octet-stream", "upsert": "true"}
            )
            print(f"Supabase storage upload response: {response}")

        # 3. Clean up the temporary local file
        os.remove(local_temp_path)
        print(f"Temporary file {local_temp_path} removed.")
        print("Pickled artifact saved successfully to Supabase Storage.")
        return storage_path
    except Exception as e:
        print(f"Error saving pickled artifact: {e}")
        if os.path.exists(local_temp_path):
            try:
                os.remove(local_temp_path)
                print(f"Cleaned up temporary file {local_temp_path} after error.")
            except OSError as rm_err:
                print(f"Error removing temporary file {local_temp_path} after error: {rm_err}")
        raise


def select_model_for_forecast(feeder_id, architecture=None, scenario=None, version=None):
    """Queries ml.Models to find the desired model metadata using an EXACT version match if provided."""
    print(f"Selecting model for Feeder {feeder_id}, Arch: {architecture}, Scenario: {scenario}, Version: {version}")
    try:
        supabase.postgrest.schema(ML_SCHEMA)
        query = supabase.schema(ML_SCHEMA).table("models").select("*").eq("feeder_id", feeder_id)

        # Add filters based on provided arguments
        if architecture:
            query = query.eq("model_architecture_type", architecture)
        if scenario:
            query = query.eq("scenario_type", scenario)
        # --- Use exact version match ---
        # if version:
        #     print(f"Filtering for EXACT version: {version}")
        #     query = query.eq("model_version", version)
        # --- End exact version match ---

        # If version is specified, we expect only one match.
        # If version is NOT specified, get the latest matching the other criteria.
        if not version:
            query = query.order("training_timestamp", desc=True).limit(1)

        response = query.execute()

        # Handle potential multiple matches if version wasn't specific enough (shouldn't happen with exact match)
        if response.data and len(response.data) > 1 and version:
            print(f"Warning: Found multiple models ({len(response.data)}) matching exact version '{version}'. Using the first one found.")

        if response.data:
            model_metadata = response.data[0]
            print(
                f"Selected Model ID: {model_metadata.get('model_id')}, Version: {model_metadata.get('model_version')}, Path Info: {model_metadata.get('model_artifact_path')}"
            )
            # Parse JSON fields
            for key in ["model_hyperparameters", "feature_engineering_config", "validation_metrics"]:
                if model_metadata.get(key) and isinstance(model_metadata[key], str):
                    try:
                        model_metadata[key] = json.loads(model_metadata[key])
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse JSON for {key}")
            # Special parsing for model_artifact_path if it's JSON (Keras case)
            if isinstance(model_metadata.get("model_artifact_path"), str) and model_metadata["model_artifact_path"].startswith("{"):
                try:
                    model_metadata["model_artifact_path"] = json.loads(model_metadata["model_artifact_path"])
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON for model_artifact_path")
            return model_metadata
        else:
            print(f"Error: No matching model found for criteria: Feeder={feeder_id}, Arch={architecture}, Scenario={scenario}, Version={version}")
            return None
    except Exception as e:
        print(f"Error selecting model metadata: {e}")
        raise


def store_forecasts(model_id, feeder_id, forecast_run_timestamp, target_date, predictions_original, target_columns):
    """Formats and inserts forecast results into ml.Forecasts."""
    print(f"Storing forecasts for Feeder {feeder_id}, Target Date: {target_date}, Model ID: {model_id}")
    if predictions_original is None or predictions_original.size == 0:
        print("Warning: No predictions provided to store.")
        return
    try:
        actual_hours = sorted([int(col.split("_Hour_")[-1]) for col in target_columns])
        if len(actual_hours) != predictions_original.shape[1]:
            raise ValueError("Mismatch between target columns and prediction shape.")
        target_timestamps = [datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=h) for h in actual_hours]
        forecast_values = predictions_original.flatten()
        if len(forecast_values) != len(target_timestamps):
            raise ValueError(f"Mismatch between number of forecast values ({len(forecast_values)}) and target timestamps ({len(target_timestamps)}).")
        records_to_insert = [
            {
                "model_id": model_id,
                "feeder_id": feeder_id,
                "forecast_run_timestamp": forecast_run_timestamp,
                "target_timestamp": ts.isoformat(),
                "forecast_value": float(forecast_values[i]),
                "actual_value": None,
            }
            for i, ts in enumerate(target_timestamps)
        ]
        supabase.postgrest.schema(ML_SCHEMA)
        response = supabase.schema(ML_SCHEMA).table("forecasts").insert(records_to_insert).execute()
        if hasattr(response, "data") and response.data:
            print(f"Successfully inserted {len(response.data)} forecast records.")
        elif hasattr(response, "error") and response.error:
            print(f"Error inserting forecasts: {response.error}")
            raise Exception(f"Failed to insert forecasts: {response.error}")
        elif not hasattr(response, "error") and not hasattr(response, "data"):
            print(f"Forecasts inserted (assumed based on response, count unknown).")
        else:
            print(f"Unknown error inserting forecasts. Response: {response}")
            raise Exception("Unknown error inserting forecasts.")
    except Exception as e:
        print(f"Error formatting or storing forecasts: {e}")
        traceback.print_exc()
        raise


def load_artifact_from_storage(artifact_path_info):
    """Downloads artifact(s) from Supabase Storage and loads them."""
    print(f"Loading artifact(s) based on path info: {artifact_path_info}")
    os.makedirs(TEMP_DIR, exist_ok=True)
    keras_model_path, scalers_pkl_path, single_pkl_path = None, None, None
    local_keras_temp_path, local_scalers_temp_path, local_single_pkl_temp_path = None, None, None
    if isinstance(artifact_path_info, dict):
        keras_model_path = artifact_path_info.get("keras_model")
        scalers_pkl_path = artifact_path_info.get("scalers_pkl")
        if not keras_model_path or not scalers_pkl_path:
            raise ValueError("Invalid artifact path info dictionary.")
        print(f"Detected separate Keras model ({keras_model_path}) and scalers ({scalers_pkl_path}).")
        local_keras_temp_path = os.path.join(TEMP_DIR, os.path.basename(keras_model_path))
        local_scalers_temp_path = os.path.join(TEMP_DIR, os.path.basename(scalers_pkl_path))
    elif isinstance(artifact_path_info, str) and artifact_path_info.endswith(".pkl"):
        single_pkl_path = artifact_path_info
        print(f"Detected single pickle artifact path: {single_pkl_path}")
        local_single_pkl_temp_path = os.path.join(TEMP_DIR, os.path.basename(single_pkl_path))
    else:
        raise ValueError(f"Unsupported artifact path info format: {artifact_path_info}")
    loaded_model, loaded_scalers_info = None, {}
    try:
        if keras_model_path:
            if not KERAS_AVAILABLE:
                raise RuntimeError("Keras artifact specified, but TensorFlow/Keras is not installed.")
            print(f"Downloading Keras model to: {local_keras_temp_path}")

            with open(local_keras_temp_path, "wb+") as f:
                res = supabase.storage.from_(STORAGE_BUCKET).download(keras_model_path)
                f.write(res)

            print("Keras model downloaded. Loading...")
            loaded_model = load_model(local_keras_temp_path, custom_objects={"NormalizeLayer": NormalizeLayer})
            print("Keras model loaded.")
        if scalers_pkl_path:
            print(f"Downloading scalers pickle to: {local_scalers_temp_path}")
            with open(local_scalers_temp_path, "wb+") as f:
                res = supabase.storage.from_(STORAGE_BUCKET).download(scalers_pkl_path)
                f.write(res)
            print("Scalers pickle downloaded. Loading...")
            with open(local_scalers_temp_path, "rb") as f:
                loaded_scalers_info = pickle.load(f)
                print("Scalers pickle loaded.")
            if not isinstance(loaded_scalers_info, dict) or not all(
                k in loaded_scalers_info for k in ["x_scaler", "y_scaler", "feature_columns", "target_columns"]
            ):
                raise TypeError("Loaded scalers pickle file does not contain expected keys.")
        if single_pkl_path:
            print(f"Downloading single pickle artifact to: {local_single_pkl_temp_path}")
            with open(local_single_pkl_temp_path, "wb+") as f:
                res = supabase.storage.from_(STORAGE_BUCKET).download(single_pkl_path)
                f.write(res)
            print("Single pickle downloaded. Loading...")
            with open(local_single_pkl_temp_path, "rb") as f:
                loaded_object = pickle.load(f)
                print("Single pickle loaded.")
            if isinstance(loaded_object, dict) and "model" in loaded_object:
                loaded_model = loaded_object["model"]
                loaded_scalers_info = loaded_object
            elif PADASIP_AVAILABLE and isinstance(loaded_object.get("rls_filters"), list):
                print("Detected RLS filters artifact.")
                return {"rls_filters": loaded_object["rls_filters"], "target_columns": loaded_object.get("target_columns")}
            else:
                raise TypeError("Loaded single pickle file has unexpected structure.")
        if loaded_model is None and "rls_filters" not in loaded_scalers_info:
            raise ValueError("Failed to load the primary model or RLS filters.")
        return {
            "model": loaded_model,
            "x_scaler": loaded_scalers_info.get("x_scaler"),
            "y_scaler": loaded_scalers_info.get("y_scaler"),
            "feature_columns": loaded_scalers_info.get("feature_columns"),
            "target_columns": loaded_scalers_info.get("target_columns"),
            "rls_filters": loaded_scalers_info.get("rls_filters"),
        }  # Include rls_filters if present
    except Exception as e:
        print(f"Error loading artifact(s): {e}")
        traceback.print_exc()
        raise
    finally:
        for temp_path in [local_keras_temp_path, local_scalers_temp_path, local_single_pkl_temp_path]:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"Cleaned up temporary file: {temp_path}")
                except OSError as rm_err:
                    print(f"Error removing temporary file {temp_path}: {rm_err}")


# Add this function (e.g., in DB_Utils.py or the main script)


def store_validation_results(model_id, feeder_id, validation_run_timestamp, y_val_actual_df, y_pred_val_original_np, target_columns):
    """
    Formats and inserts validation results (actuals and predictions) into ml.Forecasts.

    Args:
        model_id (int): The ID of the model artifact that generated the predictions.
        feeder_id (int): The Feeder ID.
        validation_run_timestamp (str): ISO format timestamp when validation was run.
        y_val_actual_df (pd.DataFrame): DataFrame of actual validation values (index=date, columns=hourly targets).
        y_pred_val_original_np (np.array): Numpy array of predicted validation values (original scale).
        target_columns (list): List of target column names (e.g., ['Net_Load_Demand_Hour_6', ...]).
    """
    print(f"Storing validation results for Model ID: {model_id}, Feeder: {feeder_id}...")

    if y_val_actual_df.empty or y_pred_val_original_np.size == 0:
        print("Warning: Empty actuals or predictions provided for validation storage.")
        return
    if y_val_actual_df.shape != y_pred_val_original_np.shape:
        print(
            f"ERROR: Shape mismatch between validation actuals ({y_val_actual_df.shape}) and predictions ({y_pred_val_original_np.shape}). Cannot store."
        )
        return
    if list(y_val_actual_df.columns) != target_columns:
        print(f"ERROR: Target columns list does not match actuals DataFrame columns.")
        # Attempt reorder? Or just error out. Erroring out is safer.
        # print("Actual Columns:", list(y_val_actual_df.columns))
        # print("Target Columns:", target_columns)
        return

    records_to_insert = []
    num_days, num_hours = y_val_actual_df.shape

    try:
        # Extract actual hours from target columns
        actual_hours = sorted([int(col.split("_Hour_")[-1]) for col in target_columns])
        if len(actual_hours) != num_hours:
            raise ValueError("Could not parse correct number of hours from target columns.")

        # Iterate through each day in the validation set
        for day_idx, target_date in enumerate(y_val_actual_df.index):
            # Ensure target_date is a date object if it's datetime
            if isinstance(target_date, pd.Timestamp):
                target_date = target_date.date()

            # Iterate through each hour predicted for that day
            for hour_idx, hour_of_day in enumerate(actual_hours):
                target_timestamp = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=hour_of_day)
                actual_value = y_val_actual_df.iloc[day_idx, hour_idx]
                forecast_value = y_pred_val_original_np[day_idx, hour_idx]

                # Skip if actual value is NaN (can happen with data gaps)
                if pd.isna(actual_value):
                    continue

                records_to_insert.append(
                    {
                        "model_id": model_id,
                        "feeder_id": feeder_id,
                        "forecast_run_timestamp": validation_run_timestamp,  # Timestamp of the validation run
                        "target_timestamp": target_timestamp.isoformat(),
                        "forecast_value": float(forecast_value) if not pd.isna(forecast_value) else None,  # Store prediction
                        "actual_value": float(actual_value),  # Store actual
                    }
                )

        if not records_to_insert:
            print("No valid records generated for validation storage.")
            return

        # Insert into database
        print(f"Inserting {len(records_to_insert)} validation records into ml.Forecasts...")
        supabase.postgrest.schema(ML_SCHEMA)
        # Consider chunking inserts if len(records_to_insert) is very large (e.g., > 1000)
        response = supabase.schema(ML_SCHEMA).table("forecasts").insert(records_to_insert).execute()

        # Check response
        if hasattr(response, "data") and response.data:
            print(f"Successfully inserted {len(response.data)} validation records.")
        elif hasattr(response, "error") and response.error:
            print(f"Error inserting validation results: {response.error}")
            # Handle potential conflicts (e.g., unique constraint violation if re-running)
            if "23505" in str(response.error):
                print("Hint: Validation results for this model/feeder/target_timestamp might already exist.")
            # Don't raise exception here, just log error during training run
        elif not hasattr(response, "error") and not hasattr(response, "data"):
            print(f"Validation results inserted (assumed based on response, count unknown).")
        else:
            print(f"Unknown error inserting validation results. Response: {response}")

    except Exception as e:
        print(f"Error formatting or storing validation results: {e}")
        traceback.print_exc()
        # Don't raise exception, allow training run to continue if possible
