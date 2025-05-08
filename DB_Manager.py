import pickle
from io import BytesIO
import sys
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from supabase import create_client, Client
import os
from datetime import datetime, timedelta
import lightgbm as lgb
import pandas as pd
import numpy as np
import base64
import tempfile
import traceback
from keras.models import Sequential, save_model, load_model

# from RLSCombiner import RLSCombiner
import json


class DatabaseManager:
    def __init__(self, tag="main"):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SECRET_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SECRET_KEY must be set as environment variables")
        self.client: Client = create_client(url, key)
        self.ML_SCHEMA = "ml"
        self.DATA_SCHEMA = "data"
        self.METADATA_SCHEMA = "metadata"
        self.trained_models_bucket = "trained-models"
        self.rls_combiners_bucket = "rls-combiners"
        self.tag = tag  # <-- NEW: Default "main" unless overridden

    def save_scaler(self, feeder_id: int, scaler: object, version: str, purpose: str, load_type: str, scenario: str):
        print("Saving scaler to Supabase...")

        # First check if scaler exists
        existing = (
            self.client.schema(self.ML_SCHEMA)
            .table("scalers")
            .select("scaler_encoded")
            .eq("feeder_id", feeder_id)
            .eq("scaler_version", version)
            .eq("purpose", purpose)
            .eq("load_type", load_type)
            .eq("scenario", scenario)
            .execute()
        )
        if existing.data:
            print("Scaler already exists. Reusing existing scaler.")
            return existing

        try:
            serialized = pickle.dumps(scaler)
            encoded = base64.b64encode(serialized).decode("utf-8")
            response = (
                self.client.schema(self.ML_SCHEMA)
                .table("scalers")
                .upsert(
                    {
                        "feeder_id": feeder_id,
                        "scaler_version": version,
                        "purpose": purpose,
                        "load_type": load_type,
                        "scenario": scenario,
                        "scaler_encoded": encoded,
                    }
                )
                .execute()
            )
            print("Scaler save successful")
            return response

        except Exception as e:
            print(f"Error saving scaler: {e}")
            raise

    def get_model_id(self, feeder_id, scenario, arch_type, version):
        response = (
            self.client.schema(self.ML_SCHEMA)
            .table("models")
            .select("model_id")
            .match(
                {
                    "feeder_id": feeder_id,
                    "scenario_type": scenario,
                    "model_architecture_type": arch_type,
                    "model_version": version,
                    "tag": self.tag,
                }
            )
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0]["model_id"]
        else:
            raise ValueError("Model ID not found!")

    def save_feeder_stats(self, feeder_id: int, scenario_type: str, train_start: str, train_end: str, feeder_stats_df: pd.DataFrame):
        """
        Save feature statistics into metadata.feeder_stats.

        feeder_stats_df must have columns: ['feature_name', 'mean', 'std', 'min', 'max']
        """
        if feeder_stats_df.empty:
            print("Warning: feeder_stats_df is empty.")
            return None

        entries = []
        for _, row in feeder_stats_df.iterrows():
            entry = {
                "feeder_id": feeder_id,
                "scenario_type": scenario_type,
                "train_start_date": train_start,
                "train_end_date": train_end,
                "feature_name": row["feature_name"],
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "min": float(row["min"]),
                "max": float(row["max"]),
            }
            entries.append(entry)

        response = (
            self.client.schema(self.METADATA_SCHEMA)
            .table("feeder_stats")
            .upsert(entries, on_conflict="feeder_id, scenario_type, train_start_date, train_end_date, feature_name")
            .execute()
        )

        print(f"Saved {len(entries)} feature statistics to metadata.feeder_stats.")
        return response

    def load_feeder_stats(self, feeder_id: int, scenario_type: str, train_start: str, train_end: str):
        """
        Load feature statistics from metadata.feeder_stats.
        """

        # print("________Test______________", feeder_id, scenario_type, train_start, train_end)
        response = (
            self.client.schema(self.METADATA_SCHEMA)
            .table("feeder_stats")
            .select("*")
            .eq("feeder_id", feeder_id)
            .eq("scenario_type", scenario_type)
            .eq("train_start_date", train_start)
            .eq("train_end_date", train_end)
            .execute()
        )

        if not response.data:
            print("No feature stats found.")
            return pd.DataFrame()

        df = pd.DataFrame(response.data)
        # print("+++++++++++++++++++++++++++")
        # print(df)
        # print("+++++++++++++++++++++++++++")
        return df.set_index("feature_name")

    def save_forecasts(
        self,
        feeder_id: int,
        version: str,
        scenario_type: str,
        model_architecture_type: str,
        forecasts_df: pd.DataFrame,
    ):
        """
        Save forecast entries into ml.forecasts table with associated tag.

        Assumes forecasts_df has:
            - DatetimeIndex as timestamps
            - 'Forecast' column
            - Optional: 'actual_value' column
        """
        try:
            if forecasts_df.empty:
                print("Warning: Provided forecast DataFrame is empty.")
                return None

            if "forecast_value" not in forecasts_df.columns:
                raise ValueError("forecast_value column is required in forecasts_df.")

            entries = []
            for ts, row in forecasts_df.iterrows():

                entry = {
                    "feeder_id": feeder_id,
                    # "scenario_type": scenario_type,
                    "model_version": version,
                    # "model_architecture_type": model_architecture_type,
                    "forecast_run_timestamp": datetime.utcnow().isoformat(),
                    "target_timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "forecast_value": float(row["forecast_value"]),
                    "tag": self.tag,
                }
                if "actual_value" in row and pd.notna(row["actual_value"]):
                    entry["actual_value"] = float(row["actual_value"])

                entry["scenario_type"] = scenario_type
                entry["model_architecture_type"] = model_architecture_type

                # print(entry)
                entries.append(entry)

            print("First entry going into upsert:", entries[0])
            # self.validate_forecast_entries(entries, required_conflict_fields=["feeder_id", "model_version", "tag", "target_timestamp"])

            # sys.exit(1)

            response = (
                self.client.schema(self.ML_SCHEMA)
                .table("forecasts")
                .upsert(entries, on_conflict="feeder_id, model_version, tag, target_timestamp")
                # .upsert(entries)
                .execute()
            )

            print(f"Saved {len(entries)} forecast entries with tag '{self.tag}'.")
            return response

        except Exception as e:
            print(f"Error saving forecasts: {e}")
            traceback.print_exc()
            raise

    def validate_forecast_entries(self, entries: list, required_conflict_fields: list):
        """
        Validate that forecast entries are properly structured before upsert.

        Args:
            entries (list of dicts): Forecast entries to insert/upsert.
            required_conflict_fields (list of str): Fields required to resolve conflicts (should match database UNIQUE constraint).

        Raises:
            ValueError: If required fields are missing or incorrectly named.
        """

        if not entries:
            raise ValueError("No forecast entries provided.")

        # Get all keys from first entry
        entry_keys = set(entries[0].keys())

        # Check all conflict fields exist
        missing_fields = [field for field in required_conflict_fields if field not in entry_keys]
        if missing_fields:
            raise ValueError(f"Missing required conflict fields in entries: {missing_fields}")

        # Basic sanity checks
        if "forecast_value" not in entry_keys:
            raise ValueError("'forecast_value' field must be present in forecast entries.")

        if "target_timestamp" not in entry_keys:
            raise ValueError("'target_timestamp' field must be present in forecast entries.")

        print(f"✅ Validation successful: Entries ready to be saved ({len(entries)} rows).")

    def load_forecasts(
        self,
        feeder_id: int,
        version: str,
        scenario_type: Optional[str] = None,
        model_architecture_type: Optional[str] = None,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        tag: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load forecast entries from ml.forecasts table based on feeder_id, version, and tag.
        Optionally filter by scenario_type, model_architecture_type, and timestamp range.
        """

        tag = tag if tag else self.tag  # Use provided tag or default to instance tag
        try:
            query = (
                self.client.schema(self.ML_SCHEMA)
                .table("forecasts")
                .select("*")
                .eq("feeder_id", feeder_id)
                .eq("model_version", version)
                .eq("tag", tag)  # Use provided tag or default to instance tag
            )

            if scenario_type:
                query = query.eq("scenario_type", scenario_type)
            if model_architecture_type:
                query = query.eq("model_architecture_type", model_architecture_type)
            if start_timestamp:
                query = query.gte("target_timestamp", start_timestamp)
            if end_timestamp:
                query = query.lte("target_timestamp", end_timestamp)

            response = query.order("target_timestamp", desc=False).execute()

            if not response.data:
                print("Warning: No forecast data found.")
                return pd.DataFrame()

            df = pd.DataFrame(response.data)
            df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])
            df = df.set_index("target_timestamp")
            print(f"Loaded {len(df)} forecast entries with tag '{tag}'.")
            return df

        except Exception as e:
            print(f"Error loading forecasts: {e}")
            traceback.print_exc()
            raise

    def save_model_to_supabase(self, model, feeder_id, arch_type, scenario, version, start_ts, end_ts, hyperparams, load_type):
        bucket = self.trained_models_bucket
        file_name = f"model_{feeder_id}_{arch_type}_{load_type}_{scenario}_{version}_{self.tag}"
        tmp_path = os.path.join(tempfile.gettempdir(), file_name)

        if arch_type in ["LSTM", "ANN"]:
            tmp_path += ".keras"
            save_model(model, tmp_path)
        else:
            tmp_path += ".pkl"
            with open(tmp_path, "wb") as f:
                pickle.dump(model, f)

        supabase_path = os.path.basename(tmp_path)

        # Try deleting old file
        try:
            self.client.storage.from_(bucket).remove([supabase_path])
            print(f"Deleted existing file: {supabase_path}")
        except Exception:
            pass

        self.client.storage.from_(bucket).upload(file=tmp_path, path=supabase_path, file_options={"content-type": "application/octet-stream"})
        os.remove(tmp_path)

        # Deactivate previous models only for this tag/version/arch
        if arch_type == "LSTM":
            self.client.schema(self.ML_SCHEMA).table("models").update({"is_active_for_forecast": False}).match(
                {
                    "feeder_id": feeder_id,
                    "scenario_type": scenario,
                    "model_architecture_type": arch_type + f"_{load_type.capitalize()}",
                    "tag": self.tag,
                }
            ).execute()

        # Insert new model
        self.client.schema(self.ML_SCHEMA).table("models").upsert(
            {
                "feeder_id": feeder_id,
                "scenario_type": scenario,
                "model_architecture_type": arch_type + f"_{load_type.capitalize()}",
                "model_version": version,
                "train_start_date": start_ts,
                "train_end_date": end_ts,
                "model_hyperparameters": hyperparams or {"version": "default"},
                "model_artifact_path": supabase_path,
                "training_timestamp": datetime.now().isoformat(),
                "validation_metrics": None,
                "test_metrics": None,
                "is_active_for_forecast": True if arch_type == "LSTM" else False,
                "tag": self.tag,
            },
            on_conflict="model_artifact_path",
        ).execute()

    def load_models_from_supabase(self, feeder_id, scenario, arch_type="LSTM"):
        bucket = self.trained_models_bucket

        def _load_single_model(load_type, arch_type):
            query = (
                self.client.schema(self.ML_SCHEMA)
                .table("models")
                .select("model_artifact_path")
                .match(
                    {
                        "feeder_id": feeder_id,
                        "scenario_type": scenario,
                        "model_architecture_type": f"{arch_type}_{load_type.capitalize()}",
                        "is_active_for_forecast": True if arch_type == "LSTM" else False,
                        "tag": self.tag,
                    }
                )
                .execute()
            )
            if not query.data:
                raise FileNotFoundError(f"No active model found for {arch_type}_{load_type} with tag {self.tag}.")

            path = query.data[0]["model_artifact_path"]
            download_path = os.path.join(tempfile.gettempdir(), path)

            with open(download_path, "wb") as f:
                f.write(self.client.storage.from_(bucket).download(path))

            _, ext = os.path.splitext(path)
            if ext == ".keras":
                return load_model(download_path)
            elif ext == ".pkl":
                with open(download_path, "rb") as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported model file extension: {ext}")

        base_model = _load_single_model("base", arch_type)
        change_model = _load_single_model("change", arch_type) if arch_type != "LightGBM" else None
        return base_model, change_model

    def get_all_feeder_ids(self):
        """Fetches all Feeder_ID values from the metadata table."""
        print("Fetching list of Feeder IDs...")
        try:
            self.client.postgrest.schema(self.METADATA_SCHEMA)
            response = self.client.schema(self.METADATA_SCHEMA).table("Feeders_Metadata").select("Feeder_ID").execute()
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
            self.client.postgrest.schema("public")  # Reset schema

    def load_scaler(self, feeder_id: int, version: str, purpose: str, load_type: str, scenario: str = "24hr"):
        if version == "latest":
            response = (
                self.client.schema(self.ML_SCHEMA)
                .table("scalers")
                .select("scaler_version, scaler_encoded")
                .eq("feeder_id", feeder_id)
                .eq("purpose", purpose)
                .eq("load_type", load_type)
                .eq("scenario", scenario)
                .order("scaler_version", desc=True)
                .limit(1)
                .execute()
            )
        else:
            response = (
                self.client.schema(self.ML_SCHEMA)
                .table("scalers")
                .select("scaler_encoded")
                .eq("feeder_id", feeder_id)
                .eq("scaler_version", version)
                .eq("load_type", load_type)
                .eq("scenario", scenario)
                .eq("purpose", purpose)
                .execute()
            )

        if response.data:
            scaler_encoded = response.data[0]["scaler_encoded"]
            scaler = pickle.loads(base64.b64decode(scaler_encoded.encode("utf-8")))

            return scaler

        print(f"Scaler not found for feeder_id={feeder_id}, version={version}, purpose={purpose}, load_type={load_type}, scenario={scenario}")
        return None

    # --- Database Interaction Functions ---
    def fetch_data(self, feeder_id, start_date, end_date):
        """Fetches combined feeder and weather data from Supabase."""
        print(f"Fetching data for Feeder {feeder_id} from {start_date} to {end_date}...")
        end_date_dt = pd.to_datetime(end_date) + timedelta(days=1)  # Include the end date in the range
        end_date_str = end_date_dt.strftime("%Y-%m-%d %H:%M:%S%z")
        try:
            response = (
                self.client.schema(self.DATA_SCHEMA)
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
                print(self.DATA_SCHEMA, feeder_id, start_date, end_date_str)
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

    def save_rls_filters(self, rls_filters, feeder_id, version, scenario, model_arch):
        """Saves the RLS filters to Supabase Storage and updates the database reference."""

        bucket = self.rls_combiners_bucket
        # Serialize + upload to Supabase Storage
        # file_path = f"rls_combiners/f_{feeder_id}_{scenario}_{version}.pkl"

        # print("Saving RLS filters to Supabase Storage...")
        # print(f"RLS filters path: {file_path}")

        serialized = pickle.dumps(rls_filters)
        encoded = base64.b64encode(serialized).decode("utf-8")  # <- JSON-safe string

        # with tempfile.NamedTemporaryFile(delete=False) as tmp:
        #     pickle.dump(rls_filters, tmp)
        #     tmp.flush()
        #     self.client.storage.from_(bucket).upload(file=tmp.name, path=file_path, file_options={"upsert": True})
        #     os.remove(tmp.name)

        # Update DB reference
        self.client.schema(self.ML_SCHEMA).table("rls_combiners").upsert(
            {
                "feeder_id": feeder_id,
                "scenario": scenario,
                "model_version": version,
                "model_arch": model_arch,
                "rls_combiner_encoded": encoded,
                "last_updated": datetime.utcnow().isoformat(),
                "tag": self.tag,
            }
        ).execute()

    def load_rls_filters(self, feeder_id, version, scenario, model_arch):
        """Loads the RLS filters from Supabase Storage."""

        bucket = self.rls_combiners_bucket

        # Check if the RLS filters exist in the database
        response = (
            self.client.schema(self.ML_SCHEMA)
            .table("rls_combiners")
            .select("rls_combiner_encoded")
            .eq("feeder_id", feeder_id)
            .eq("model_version", version)
            .eq("model_arch", model_arch)
            .eq("scenario", scenario)
            .eq("tag", self.tag)
            .limit(1)
            .execute()
        )

        if response.data:
            rls_combiner_encoded = response.data[0]["rls_combiner_encoded"]
            rls_combiner = pickle.loads(base64.b64decode(rls_combiner_encoded.encode("utf-8")))

            return rls_combiner
        return None


# class DatabaseManager:
#     def __init__(self):
#         url = os.environ.get("SUPABASE_URL")
#         key = os.environ.get("SUPABASE_SECRET_KEY")
#         if not url or not key:
#             raise ValueError("SUPABASE_URL and SUPABASE_SECRET_KEY must be set as environment variables")
#         self.client: Client = create_client(url, key)
#         self.ML_SCHEMA = "ml"
#         self.DATA_SCHEMA = "data"
#         self.METADATA_SCHEMA = "metadata"
#         self.trained_models_bucket = "trained-models"
#         self.rls_combiners_bucket = "rls-combiners"


#     def save_scaler(self, feeder_id: int, scaler: object, version: str, purpose: str, load_type: str, scenario: str):
#         print("Saving scaler to Supabase...")
#         try:
#             serialized = pickle.dumps(scaler)
#             encoded = base64.b64encode(serialized).decode("utf-8")  # <- JSON-safe string
#             response = (
#                 self.client.schema(self.ML_SCHEMA)
#                 .table("scalers")
#                 .upsert(
#                     {
#                         "feeder_id": feeder_id,
#                         "scaler_version": version,
#                         "purpose": purpose,
#                         "load_type": load_type,
#                         "scenario": scenario,
#                         "scaler_encoded": encoded,
#                     },
#                     # on_conflict=["feeder_id", "scaler_version", "purpose", "load_type"],
#                 )
#                 .execute()
#             )

#             print("Scaler save successful")
#             return response

#         except Exception as e:
#             print(f"Error saving scaler: {e}")
#             raise e


#     # ----------------- Save Model to Supabase ----------------- #
#     def save_model_to_supabase(self, model, feeder_id, arch_type, scenario, version, start_ts, end_ts, hyperparams, load_type):
#         # db = DatabaseManager()
#         bucket = "trained-models"

#         file_name = f"model_{feeder_id}_{arch_type}_{load_type}_{scenario}_{version}"
#         tmp_path = os.path.join(tempfile.gettempdir(), file_name)

#         if arch_type == "LSTM" or arch_type == "ANN":
#             tmp_path += ".keras"
#             save_model(model, tmp_path)
#             print(tmp_path)
#             # print("Checking if model is saved correctly", os.listdir(tmp_path))
#             if os.path.exists(tmp_path):
#                 print("Model exists.")
#             else:
#                 print("Model not saved.")
#         else:
#             tmp_path += ".pkl"
#             with open(tmp_path, "wb") as f:
#                 pickle.dump(model, f)

#         supabase_path = os.path.basename(tmp_path)

#         print("Temp file path:", tmp_path)
#         print("Supabase path:", supabase_path)

#         # First try deleting (safe even if file does not exist)
#         try:
#             self.client.storage.from_(bucket).remove([supabase_path])
#             print(f"Deleted existing file: {supabase_path}")
#         except Exception as e:
#             print(f"No file to delete or error during delete: {e}")

#         self.client.storage.from_(bucket).upload(file=tmp_path, path=supabase_path, file_options={"content-type": "application/octet-stream"})
#         print("Model uploaded to Supabase Storage.")

#         os.remove(tmp_path)  # Clean up temp file

#         self.client.schema(self.ML_SCHEMA).table("models").update({"is_active_for_forecast": False}).match(
#             {
#                 "feeder_id": feeder_id,
#                 "scenario_type": scenario,
#                 "model_architecture_type": arch_type + f"_{load_type.capitalize()}",
#             }
#         ).execute()

#         self.client.schema(self.ML_SCHEMA).table("models").upsert(
#             {
#                 "feeder_id": feeder_id,
#                 "scenario_type": scenario,
#                 "model_architecture_type": arch_type + f"_{load_type.capitalize()}",
#                 "model_version": version,
#                 "train_data_start_timestamp": start_ts,
#                 "train_end_date": end_ts,
#                 "model_hyperparameters": hyperparams or {"version": "default"},
#                 "model_artifact_path": supabase_path,
#                 "training_timestamp": datetime.now().isoformat(),
#                 "validation_metrics": None,
#                 "test_metrics": None,
#                 "is_active_for_forecast": True,
#             }
#         ).execute()
#         print(f"Model ({arch_type}_{load_type}) saved to Supabase Storage: {supabase_path}")

#     # ----------------- Load Models from Supabase ----------------- #
#     def load_models_from_supabase(self, feeder_id, scenario, arch_type="LSTM"):
#         # db = DatabaseManager()
#         bucket = self.trained_models_bucket

#         def _load_single_model(load_type):
#             query = (
#                 self.client.schema(self.ML_SCHEMA)
#                 .table("models")
#                 .select("model_artifact_path")
#                 .match(
#                     {
#                         "feeder_id": feeder_id,
#                         "scenario_type": scenario,
#                         "model_architecture_type": f"{arch_type}_{load_type.capitalize()}",
#                         "is_active_for_forecast": True,
#                     }
#                 )
#                 .execute()
#             )
#             if not query.data:
#                 raise FileNotFoundError(f"No active model found for {arch_type}_{load_type}.")

#             path = query.data[0]["model_artifact_path"]
#             print(f"Loading model from path: {path}")
#             _, ext = os.path.splitext(path)
#             download_path = os.path.join(tempfile.gettempdir(), path)

#             with open(download_path, "wb") as f:
#                 f.write(self.client.storage.from_(bucket).download(path))

#             if ext == ".keras":
#                 return load_model(download_path)
#             elif ext == ".pkl":
#                 with open(download_path, "rb") as f:
#                     return pickle.load(f)
#             else:
#                 raise ValueError(f"Unsupported model file extension: {ext}")

#         base_model = _load_single_model("base")
#         change_model = _load_single_model("change") if arch_type != "LightGBM" else None
#         return base_model, change_model


# def save_forecasts(
#     self,
#     feeder_id: int,
#     version: str,
#     model_type: str,
#     timestamps,
#     predictions,
#     actuals,
#     scenario_type: str = "24hr",
#     experiment_tag: str = "Baseline",
# ):
#     try:
#         self.client.postgrest.rpc("create_forecast_predictions_table_if_not_exists").execute()
#     except Exception:
#         pass
#     entries = [
#         {
#             "feeder_id": feeder_id,
#             "version": version,
#             "model_type": model_type,
#             "scenario_type": scenario_type,
#             "experiment_tag": experiment_tag,
#             "timestamp": str(ts),
#             "predicted_value": float(pred),
#             "actual_value": float(act),
#         }
#         for ts, pred, act in zip(timestamps, predictions, actuals)
#     ]
#     self.client.table("forecast_predictions").insert(entries).execute()

# def save_tuning_results(
#     self,
#     feeder_id: int,
#     version: str,
#     model_type: str,
#     best_params: dict,
#     metric_name: str,
#     metric_value: float,
#     experiment_tag: str = "HPTuned_LSTM_RLS",
# ):
#     try:
#         self.client.postgrest.rpc("create_model_tuning_results_table_if_not_exists").execute()
#     except Exception:
#         pass
#     self.client.table("model_tuning_results").insert(
#         {
#             "feeder_id": feeder_id,
#             "version": version,
#             "model_type": model_type,
#             "best_params": json.dumps(best_params),
#             "metric_name": metric_name,
#             "metric_value": metric_value,
#             "experiment_tag": experiment_tag,
#         }
#     ).execute()

# def train_lightgbm_baseline(self, X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
#     train_data = lgb.Dataset(X_train, label=y_train)
#     params = {"objective": "regression", "metric": "rmse", "verbosity": -1}
#     model = lgb.train(params, train_data, num_boost_round=100)
#     return model

# def run_experiment_lightgbm(
#     self,
#     feeder_id: int,
#     X_train,
#     y_train,
#     X_test,
#     y_test,
#     timestamps,
#     version: str,
#     scenario_type: str = "24hr",
#     experiment_tag: str = "LightGBM_Baseline",
# ):
#     model = self.train_lightgbm_baseline(X_train, y_train)
#     y_pred = model.predict(X_test)
#     self.save_forecasts(
#         feeder_id=feeder_id,
#         version=version,
#         model_type="LightGBM_Baseload",
#         timestamps=timestamps,
#         predictions=y_pred,
#         actuals=y_test,
#         scenario_type=scenario_type,
#         experiment_tag=experiment_tag,
#     )

# def run_experiment_ann_rls(
#     self,
#     feeder_id: int,
#     y_base_pred,
#     y_change_pred,
#     y_actuals,
#     timestamps,
#     version: str,
#     scenario_type: str = "24hr",
#     experiment_tag: str = "ANN_Architecture",
# ):
#     forecasts_matrix = np.stack([y_base_pred, y_change_pred], axis=1)
#     rls = RLSCombiner(num_models=2)
#     rls.fit(forecasts_matrix, y_actuals)
#     y_rls_pred = rls.predict(forecasts_matrix)
#     self.save_forecasts(
#         feeder_id=feeder_id,
#         version=version,
#         model_type="ANN_RLS_Combined",
#         timestamps=timestamps,
#         predictions=y_rls_pred,
#         actuals=y_actuals,
#         scenario_type=scenario_type,
#         experiment_tag=experiment_tag,
#     )

# def run_experiment_daynight_split(
#     self,
#     feeder_id: int,
#     model_type: str,
#     y_pred_day,
#     y_true_day,
#     timestamps_day,
#     y_pred_night,
#     y_true_night,
#     timestamps_night,
#     version: str,
#     experiment_tag: str,
# ):
#     self.save_forecasts(
#         feeder_id=feeder_id,
#         version=version,
#         model_type=model_type,
#         timestamps=timestamps_day,
#         predictions=y_pred_day,
#         actuals=y_true_day,
#         scenario_type="daytime",
#         experiment_tag=experiment_tag,
#     )
#     self.save_forecasts(
#         feeder_id=feeder_id,
#         version=version,
#         model_type=model_type,
#         timestamps=timestamps_night,
#         predictions=y_pred_night,
#         actuals=y_true_night,
#         scenario_type="nighttime",
#         experiment_tag=experiment_tag,
#     )
