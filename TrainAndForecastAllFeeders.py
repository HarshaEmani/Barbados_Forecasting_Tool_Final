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
from Trainer_Utils_2 import run_training
from ForecasterTool import ForecasterTool
import shutil
import traceback
from DB_Manager import DatabaseManager
import plotly.express as px
from dotenv import load_dotenv, find_dotenv

KERAS_AVAILABLE = True
PADASIP_AVAILABLE = True
np.set_printoptions(suppress=True)
load_dotenv()
print("Env file found at location: ", find_dotenv())


TRAIN_START_DATE = "2024-01-01 00:00:00+00"
TRAIN_END_DATE = "2024-05-31 23:59:59+00"
VALIDATION_START_DATE = "2024-06-01 00:00:00+00"
VALIDATION_END_DATE = "2024-06-30 23:59:59+00"
TEST_START_DATE = "2024-07-01 00:00:00+00"
TEST_END_DATE = "2024-07-30 23:59:59+00"
BASE_MODEL_VERSION_PREFIX = "v2.0.9_HP_Tuning"  # Prefix for model versioning

# --- Define Architectures and Scenarios to Run ---

# List all model architectures defined in your run_training function
ARCHITECTURES_TO_RUN = [
    # "LightGBM",
    # "ANN_Baseload",
    # "ANN_Change_in_Load",
    # "LSTM_Baseload",
    # "LSTM_Change_in_Load",
    # 'ANN_RLS_Combined',
    # "LSTM_RLS_Combined",
    # 'Final_RLS_Combined'
    "LSTM",
    # "ANN",
]

# SCENARIOS_TO_RUN = ["24hr", "Day", "Night"]
TRAIN_TYPE = "Full"  # Options: "Full" or "Split"
SCENARIOS_TO_RUN = ["24hr"]  # Options: "24hr" or "Day" or "Night"


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Automated Training Run ---")

    # 1. Clear Storage
    # clear_storage_directory(supabase, STORAGE_BUCKET, "models/")

    db = DatabaseManager()
    # 2. Get Feeder IDs
    feeder_ids_to_process = db.get_all_feeder_ids()
    if not feeder_ids_to_process:
        print("No Feeder IDs found. Exiting.")
        sys.exit(1)

    run_counter = 0
    error_counter = 0

    # feeder_ids_to_process = feeder_ids_to_process[:1]  # For testing, limit to the first feeder ID

    print(f"TESTING: Feeder IDs to process: {feeder_ids_to_process}")

    for feeder_id in feeder_ids_to_process:
        for architecture in ARCHITECTURES_TO_RUN:
            if TRAIN_TYPE == "Full":
                SCENARIOS_TO_RUN = ["24hr"]
            elif TRAIN_TYPE == "Split":
                SCENARIOS_TO_RUN = ["Day", "Night"]
            else:
                raise ValueError("Invalid TRAIN_TYPE. Must be 'Full' or 'Split'.")

            # 3. Loop through Feeders, Architectures, and Scenarios
            total_runs = len(feeder_ids_to_process) * len(ARCHITECTURES_TO_RUN) * len(SCENARIOS_TO_RUN)

            print(f"\nStarting training loops for {total_runs} total combinations...")

            print(f"Feeder ID: {feeder_id}, Architecture: {architecture}, Scenarios: {SCENARIOS_TO_RUN}")

            for scenario in SCENARIOS_TO_RUN:
                run_counter += 1
                print(f"\n--- ({run_counter}/{total_runs}) Processing: Feeder={feeder_id}, Arch={architecture}, Scenario={scenario} ---")

                # Generate a unique version tag for this specific run
                timestamp_version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                dynamic_version = f"{BASE_MODEL_VERSION_PREFIX}"

                try:
                    # Call the main training function
                    forecaster_tool = ForecasterTool(
                        feeder_id=feeder_id, scenario=scenario, version=dynamic_version, model_arch=architecture, is_hyperparameter_search=True
                    )

                    if os.path.exists("tuner_dir"):
                        shutil.rmtree("tuner_dir")

                    # Run either training or forecasting based on the workflow

                    forecaster_tool.run_training(train_start_date=TRAIN_START_DATE, train_end_date=TRAIN_END_DATE)
                    forecaster_tool.train_start_date = TRAIN_START_DATE
                    forecaster_tool.train_end_date = TRAIN_END_DATE

                    forecaster_tool.forecast_range(
                        forecast_start_date="2024-05-20",
                        forecast_end_date=TEST_END_DATE,
                        new_rls=True,
                        save_daily_rls=True,
                    )

                    # run_training(
                    #     feeder_id=feeder_id,
                    #     model_arch=architecture,
                    #     scenario=scenario,
                    #     version=dynamic_version,  # Use the unique version
                    #     train_start=TRAIN_START_DATE,
                    #     train_end=TRAIN_END_DATE,
                    #     tag=f"exp_{architecture}" if architecture == "LightGBM" or architecture == "ANN" else "main",
                    #     # val_start=VALIDATION_START_DATE,
                    #     # val_end=VALIDATION_END_DATE,
                    # )
                    print(f"--- Successfully completed run ({run_counter}/{total_runs}) ---")

                except Exception as e:
                    error_counter += 1
                    print(f"!!! ERROR during run ({run_counter}/{total_runs}) for Feeder={feeder_id}, Arch={architecture}, Scenario={scenario} !!!")
                    print(f"Error message: {e}")
                    traceback.print_exc()
                    print(f"--- Skipping to next combination ---")
                    continue  # Continue to the next iteration even if one fails

    print("\n--- Automated Training Run Finished ---")
    print(f"Total combinations processed: {run_counter}")
    print(f"Successful runs (estimated): {run_counter - error_counter}")
    print(f"Failed runs: {error_counter}")
