# import tensorflow as tf
# from keras.models import Sequential, save_model, load_model
# from keras.layers import Dense, LSTM, Input, Dropout
# from keras.callbacks import EarlyStopping
# from padasip.filters import FilterRLS
# from sklearn.multioutput import MultiOutputRegressor
# from lightgbm import LGBMRegressor
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta, timezone
# import json
# import pickle
# import os
# import sys
# import argparse
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from Trainer_Utils import run_training
# from DB_Utils import fetch_data, save_pickle_artifact, log_model_metadata, train
# from supabase import create_client, Client
# import traceback
# import plotly.express as px
# from dotenv import load_dotenv, find_dotenv

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
    from Trainer_Utils import run_training
    from DB_Utils import fetch_data, save_pickle_artifact, log_model_metadata, get_all_feeder_ids
    from supabase import create_client, Client
    import traceback
    import plotly.express as px
    from dotenv import load_dotenv, find_dotenv

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
MODEL_VERSION = "v1.2_Final_Forecasting"  # Updated version
TRAIN_START_DATE = "2024-01-01 00:00:00+00"
TRAIN_END_DATE = "2024-05-31 23:59:59+00"
VALIDATION_START_DATE = "2024-06-01 00:00:00+00"
VALIDATION_END_DATE = "2024-06-30 23:59:59+00"
DAY_HOURS = list(range(6, 20 + 1))
NIGHT_HOURS = list(range(0, 6)) + list(range(21, 24))
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
TEMP_DIR = os.path.join(script_dir, "tmp")
BASE_MODEL_VERSION_PREFIX = "v1.1_Final_Forecasting"  # Prefix for model versioning

# --- Define Architectures and Scenarios to Run ---
# List all model architectures defined in your run_training function
ARCHITECTURES_TO_RUN = [
    "LightGBM_Baseline",
    # "ANN_Baseload",
    # "ANN_Change_in_Load",
    "LSTM_Baseload",
    "LSTM_Change_in_Load",
    # 'ANN_RLS_Combined',
    "LSTM_RLS_Combined",
    # 'Final_RLS_Combined'
]

# SCENARIOS_TO_RUN = ["24hr", "Day", "Night"]
SCENARIOS_TO_RUN = ["24hr"]


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Automated Training Run ---")

    # 1. Clear Storage (Optional but requested)
    # Uncomment the next line to activate storage clearing
    # clear_storage_directory(supabase, STORAGE_BUCKET, "models/")

    # 2. Get Feeder IDs
    feeder_ids_to_process = get_all_feeder_ids(supabase)
    if not feeder_ids_to_process:
        print("No Feeder IDs found. Exiting.")
        sys.exit(1)

    # 3. Loop through Feeders, Architectures, and Scenarios
    total_runs = len(feeder_ids_to_process) * len(ARCHITECTURES_TO_RUN) * len(SCENARIOS_TO_RUN)
    run_counter = 0
    error_counter = 0

    print(f"\nStarting training loops for {total_runs} total combinations...")

    for feeder_id in feeder_ids_to_process:
        for architecture in ARCHITECTURES_TO_RUN:
            for scenario in SCENARIOS_TO_RUN:
                run_counter += 1
                print(f"\n--- ({run_counter}/{total_runs}) Processing: Feeder={feeder_id}, Arch={architecture}, Scenario={scenario} ---")

                # Generate a unique version tag for this specific run
                timestamp_version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                dynamic_version = f"{BASE_MODEL_VERSION_PREFIX}_{timestamp_version}"

                try:
                    # Call the main training function
                    run_training(
                        feeder_id=feeder_id,
                        model_arch=architecture,
                        scenario=scenario,
                        version=dynamic_version,  # Use the unique version
                        train_start=TRAIN_START_DATE,
                        train_end=TRAIN_END_DATE,
                        val_start=VALIDATION_START_DATE,
                        val_end=VALIDATION_END_DATE,
                    )
                    print(f"--- Successfully completed run ({run_counter}/{total_runs}) ---")

                    # sys.exit(0)  # Exit after the first successful run for testing purposes

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
