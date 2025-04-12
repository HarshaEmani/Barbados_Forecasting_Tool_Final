try:
    # import tensorflow as tf
    # from keras.models import Sequential, save_model, load_model
    # from keras.layers import Dense, LSTM, Input, Dropout
    # from keras.callbacks import EarlyStopping
    # from padasip.filters import FilterRLS
    # from sklearn.multioutput import MultiOutputRegressor
    # from lightgbm import LGBMRegressor
    import pandas as pd
    import numpy as np

    # from datetime import datetime, timedelta, timezone
    # import json
    # import pickle
    import os
    import sys

    # import argparse
    # from sklearn.preprocessing import MinMaxScaler
    # from sklearn.metrics import mean_absolute_error, mean_squared_error
    from Trainer_Utils import run_training
    from supabase import create_client, Client

    # import traceback
    # import plotly.express as px
    # from dotenv import load_dotenv, find_dotenv

    KERAS_AVAILABLE = True
    PADASIP_AVAILABLE = True

    np.set_printoptions(suppress=True)

    # load_dotenv()
    # print("Env file found at location: ", find_dotenv())

except ImportError:
    print("TensorFlow/Keras not found. Keras models cannot be trained/saved natively.")
    KERAS_AVAILABLE = False
    # Define dummy classes if needed for type checking, though not strictly necessary here
    # Sequential, save_model, load_model = object, lambda x, y: None, lambda x: None
    # Dense, LSTM, Input, Dropout, EarlyStopping = object, object, object, object, object

    print("padasip not found. RLS filters cannot be loaded/used.")
    PADASIP_AVAILABLE = False
    FilterRLS = object


# --- Configuration & Constants --- (Same as before)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SECRET_KEY")

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


# --- Example Execution --- (Unchanged)
if __name__ == "__main__":
    if not SUPABASE_URL or not SUPABASE_KEY or "YOUR_SUPABASE_URL" in SUPABASE_URL:
        print("Error: SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
        sys.exit(1)
    else:
        for i in range(1, 2):
            run_training(
                feeder_id=i,
                model_arch="Final_RLS_Combined",  # Example: Train a Keras model
                scenario="24hr",
                version=MODEL_VERSION,
                train_start=TRAIN_START_DATE,
                train_end=TRAIN_END_DATE,
                val_start=VALIDATION_START_DATE,
                val_end=VALIDATION_END_DATE,
            )
        # run_training(
        #     feeder_id=FEEDER_ID_TO_TRAIN,
        #     model_arch='LightGBM_Baseline', # Example: Train a non-Keras model
        #     scenario='24hr',
        #     version=MODEL_VERSION,
        #     train_start=TRAIN_START_DATE,
        #     train_end=TRAIN_END_DATE,
        #     val_start=VALIDATION_START_DATE,
        #     val_end=VALIDATION_END_DATE,
        # )
