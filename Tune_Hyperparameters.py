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
    from functools import partial # For passing args to optuna objective

    # Core ML/Data Libraries
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error # For evaluation metric
    from sklearn.multioutput import MultiOutputRegressor

    # Model Libraries
    from lightgbm import LGBMRegressor
    import tensorflow as tf
    from keras.models import Sequential, save_model, load_model
    from keras.layers import Dense, LSTM, Input, Dropout, Lambda, Layer
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import keras.backend as K

    # Optuna for HPO
    import optuna

    # Supabase & Environment
    from supabase import create_client, Client
    from dotenv import load_dotenv, find_dotenv

    # Your Utility Modules (Ensure these are accessible)
    from DB_Utils import (
        fetch_data,
        select_model_for_forecast,
        load_artifact_from_storage,
        NormalizeLayer # Assuming this is in DB_Utils
    )
    from Trainer_Utils import ( # Import necessary functions from Trainer_Utils
         prepare_daily_vectors,
         feature_engineer_and_scale, # Need this for data prep
         convert_change_in_load_to_base_load # Needed if evaluating change models
         # Add any other required functions if they were defined in Trainer_Utils
    )
    # Assuming get_all_feeder_ids is also in DB_Utils or Trainer_Utils
    from DB_Utils import get_all_feeder_ids # Or Trainer_Utils.get_all_feeder_ids

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow INFO/WARNING logs
    KERAS_AVAILABLE = True
    PADASIP_AVAILABLE = False # RLS tuning not implemented here

    np.set_printoptions(suppress=True)
    load_dotenv()
    print("Env file found at location: ", find_dotenv())

except ImportError as e:
    print(f"ERROR: Failed to import necessary libraries: {e}")
    print("Check installations: pandas, numpy, sklearn, supabase, dotenv, tensorflow, lightgbm, optuna")
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
METADATA_SCHEMA = "metadata"
STORAGE_BUCKET = "models"

# --- Date Ranges ---
# Need training data to retrain models with new HPs
TRAIN_START_DATE = "2024-01-01 00:00:00+00"
TRAIN_END_DATE = "2024-05-31 23:59:59+00"
# Need validation data to evaluate HPs
VALIDATION_START_DATE = "2024-06-01 00:00:00+00"
VALIDATION_END_DATE = "2024-06-30 23:59:59+00"

# --- Tuning Configuration ---
N_TRIALS = 25 # Number of Optuna trials per model/scenario
OPTUNA_METRIC = 'mae' # Metric to optimize (on scaled data) - 'mae' or 'mse'
OPTUNA_DIRECTION = 'minimize' # Minimize MAE/MSE

# --- Architectures and Scenarios to Tune ---
# Focus on base models, exclude RLS combiners for standard HPO
ARCHITECTURES_TO_TUNE = [
    'LightGBM_Baseline',
    'ANN_Baseload',
    'ANN_Change_in_Load',
    'LSTM_Baseload',
    'LSTM_Change_in_Load',
]
SCENARIOS_TO_TUNE = ['24hr', 'Day', 'Night']


# --- Model Building Functions (Needed for Optuna) ---

def build_lgbm_model(trial, n_outputs):
    """Builds an LGBM model with hyperparameters suggested by Optuna."""
    lgbm_params = {
        'objective': 'regression_l1', # MAE objective for LGBM
        'metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 60),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True), # L1 reg
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True), # L2 reg
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1 # Suppress LGBM verbosity during tuning
    }
    base_estimator = LGBMRegressor(**lgbm_params)
    multioutput_model = MultiOutputRegressor(base_estimator, n_jobs=-1)
    return multioutput_model

def build_ann_model(trial, n_inputs, n_outputs):
    """Builds an ANN model with hyperparameters suggested by Optuna."""
    n_layers = trial.suggest_int('n_layers', 1, 3)
    model = Sequential()
    model.add(Input(shape=(n_inputs,)))
    # Input normalization layer (optional but good practice)
    # model.add(tf.keras.layers.Normalization(axis=-1)) # Or use your custom layer if needed

    for i in range(n_layers):
        units = trial.suggest_int(f'units_layer_{i}', 32, 256, log=True)
        model.add(Dense(units, activation='relu')) # Consider other activations? 'sigmoid'?
        dropout_rate = trial.suggest_float(f'dropout_layer_{i}', 0.1, 0.5)
        model.add(Dropout(dropout_rate))

    model.add(Dense(n_outputs)) # Linear output for regression is common
    # Output denormalization layer (optional)

    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error', metrics=['mae'])
    return model

def build_lstm_model(trial, n_timesteps, n_features, n_outputs):
    """Builds an LSTM model with hyperparameters suggested by Optuna."""
    n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 2)
    lstm_units = trial.suggest_int('lstm_units', 32, 128, log=True)
    n_dense_layers = trial.suggest_int('n_dense_layers', 0, 2) # Optional dense layers after LSTM

    model = Sequential()
    model.add(Input(shape=(n_timesteps, n_features))) # Shape (1, n_total_features)
    # Input normalization (optional)

    for i in range(n_lstm_layers):
        return_sequences = (i < n_lstm_layers - 1) # Return sequences for all but last LSTM layer
        model.add(LSTM(lstm_units, return_sequences=return_sequences))
        dropout_rate = trial.suggest_float(f'lstm_dropout_{i}', 0.1, 0.5)
        model.add(Dropout(dropout_rate))

    for i in range(n_dense_layers):
        dense_units = trial.suggest_int(f'dense_units_{i}', 16, 64, log=True)
        model.add(Dense(dense_units, activation='relu'))
        dropout_rate = trial.suggest_float(f'dense_dropout_{i}', 0.1, 0.5)
        model.add(Dropout(dropout_rate))

    model.add(Dense(n_outputs)) # Linear output
    # Output denormalization (optional)

    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error', metrics=['mae'])
    return model


# --- Optuna Objective Function ---
def objective(trial, architecture, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled):
    """Objective function for Optuna to minimize."""
    print(f"\n--- Optuna Trial {trial.number} for {architecture} ---")
    model = None
    is_keras = architecture.startswith('ANN') or architecture.startswith('LSTM')

    try:
        # 1. Build Model with Trial Hyperparameters
        if architecture == 'LightGBM_Baseline':
            model = build_lgbm_model(trial, n_outputs=y_train_scaled.shape[1])
        elif architecture.startswith('ANN'):
            model = build_ann_model(trial, n_inputs=X_train_scaled.shape[1], n_outputs=y_train_scaled.shape[1])
        elif architecture.startswith('LSTM'):
            # Reshape X for LSTM: (samples, timesteps=1, features)
            X_train_lstm = X_train_scaled.values.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_val_lstm = X_val_scaled.values.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
            model = build_lstm_model(trial, n_timesteps=1, n_features=X_train_scaled.shape[1], n_outputs=y_train_scaled.shape[1])
        else:
            raise ValueError(f"Unsupported architecture for tuning: {architecture}")

        # 2. Train Model
        print(f"Trial {trial.number}: Training model...")
        if is_keras:
            callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)] # Faster patience for HPO
            history = model.fit(
                X_train_lstm if architecture.startswith('LSTM') else X_train_scaled,
                y_train_scaled,
                validation_data=(X_val_lstm if architecture.startswith('LSTM') else X_val_scaled, y_val_scaled),
                epochs=50, # Limit epochs for faster tuning
                batch_size=32,
                callbacks=callbacks,
                verbose=0 # Suppress Keras fit verbosity
            )
            # Use the best validation loss from history as the metric
            # score = min(history.history['val_loss']) # Use MSE if loss is MSE
            score = min(history.history.get(f'val_{OPTUNA_METRIC}', [float('inf')])) # Use MAE if tracked

        else: # LightGBM
            model.fit(X_train_scaled, y_train_scaled)
            # 3. Evaluate Model on Validation Set (Scaled)
            print(f"Trial {trial.number}: Evaluating model...")
            y_pred_val_scaled = model.predict(X_val_scaled)
            # Calculate MAE on scaled data
            score = mean_absolute_error(y_val_scaled, y_pred_val_scaled)

        print(f"Trial {trial.number}: Validation Score ({OPTUNA_METRIC}, scaled) = {score:.6f}")

        # Handle NaN/Inf scores
        if np.isnan(score) or np.isinf(score):
             print(f"Trial {trial.number}: Invalid score detected ({score}). Reporting high value.")
             return float('inf') # Return high value for Optuna to discard

        return score

    except Exception as e:
        print(f"!!! Trial {trial.number} FAILED: {e}")
        traceback.print_exc()
        # Report a high score or prune if Optuna supports it
        # trial.report(float('inf'), step=0) # Example if using reporting
        # raise optuna.TrialPruned() # Example if pruning
        return float('inf') # Return a high value to indicate failure


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Trained Models")
    # Allow specifying subsets for testing
    parser.add_argument("--feeders", type=str, help="Comma-separated list of Feeder IDs to tune (e.g., 1,5,10)", default=None)
    parser.add_argument("--architectures", type=str, help=f"Comma-separated list of architectures to tune (default: {','.join(ARCHITECTURES_TO_TUNE)})", default=None)
    parser.add_argument("--scenarios", type=str, help=f"Comma-separated list of scenarios to tune (default: {','.join(SCENARIOS_TO_TUNE)})", default=None)
    parser.add_argument("--n_trials", type=int, help=f"Number of Optuna trials per combination (default: {N_TRIALS})", default=N_TRIALS)

    args = parser.parse_args()

    # --- Determine Feeders, Architectures, Scenarios ---
    if args.feeders:
        try: feeder_ids_to_process = [int(f.strip()) for f in args.feeders.split(',')]
        except ValueError: print("Error: Invalid feeder list format."); sys.exit(1)
    else:
        feeder_ids_to_process = get_all_feeder_ids(supabase)

    if args.architectures:
        archs_to_process = [a.strip() for a in args.architectures.split(',')]
        # Validate against allowed list
        invalid_archs = set(archs_to_process) - set(ARCHITECTURES_TO_TUNE)
        if invalid_archs: print(f"Error: Invalid architectures specified: {invalid_archs}"); sys.exit(1)
    else:
        archs_to_process = ARCHITECTURES_TO_TUNE

    if args.scenarios:
        scens_to_process = [s.strip() for s in args.scenarios.split(',')]
        invalid_scens = set(scens_to_process) - set(SCENARIOS_TO_TUNE)
        if invalid_scens: print(f"Error: Invalid scenarios specified: {invalid_scens}"); sys.exit(1)
    else:
        scens_to_process = SCENARIOS_TO_TUNE

    n_trials = args.n_trials

    if not feeder_ids_to_process: print("No Feeder IDs to process. Exiting."); sys.exit(1)

    print(f"--- Starting Hyperparameter Tuning ---")
    print(f"Feeders: {feeder_ids_to_process}")
    print(f"Architectures: {archs_to_process}")
    print(f"Scenarios: {scens_to_process}")
    print(f"Optuna Trials per combination: {n_trials}")

    # --- Fetch Data (Once per feeder) ---
    print("\nFetching training and validation data...")
    all_train_data = {}
    all_val_data = {}
    fetch_train_start = (pd.to_datetime(TRAIN_START_DATE) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S%z")
    fetch_val_start = (pd.to_datetime(VALIDATION_START_DATE) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S%z")
    fetch_val_end = (pd.to_datetime(VALIDATION_END_DATE) + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S%z") # Exclusive end

    for feeder_id in feeder_ids_to_process:
        print(f"Fetching data for Feeder {feeder_id}...")
        all_train_data[feeder_id] = fetch_data(feeder_id, fetch_train_start, TRAIN_END_DATE)
        all_val_data[feeder_id] = fetch_data(feeder_id, fetch_val_start, fetch_val_end)
        if all_train_data[feeder_id].empty or all_val_data[feeder_id].empty:
             print(f"ERROR: Insufficient data for Feeder {feeder_id}. Skipping this feeder.")
             # Remove feeder if data is missing
             # feeder_ids_to_process.remove(feeder_id) # Careful modifying list while iterating

    # Filter out feeders with missing data
    feeders_with_data = [f_id for f_id in feeder_ids_to_process if not all_train_data[f_id].empty and not all_val_data[f_id].empty]
    if not feeders_with_data:
         print("No feeders have sufficient data. Exiting.")
         sys.exit(1)

    # --- Main Tuning Loop ---
    best_params_all = {} # Dictionary to store results

    for feeder_id in feeders_with_data:
        print(f"\n===== Tuning for Feeder {feeder_id} =====")
        train_df_raw = all_train_data[feeder_id]
        val_df_raw = all_val_data[feeder_id]

        for architecture in archs_to_process:
            # Skip RLS tuning for now
            if "RLS" in architecture:
                 print(f"\n--- Skipping RLS architecture '{architecture}' for standard HPO ---")
                 continue

            for scenario in scens_to_process:
                print(f"\n--- Tuning: Arch={architecture}, Scenario={scenario} ---")

                try:
                    # 1. Load Latest Existing Model Artifact for Scalers/Columns
                    print("Loading existing model artifact to get scalers/columns...")
                    latest_metadata = select_model_for_forecast(feeder_id, architecture, scenario, version=None) # Get latest
                    if not latest_metadata:
                         print(f"Warning: No existing model found for Feeder={feeder_id}, Arch={architecture}, Scenario={scenario}. Cannot get scalers. Skipping.")
                         continue
                    loaded_artifact = load_artifact_from_storage(latest_metadata['model_artifact_path'])
                    x_scaler = loaded_artifact.get('x_scaler')
                    y_scaler = loaded_artifact.get('y_scaler')
                    # feature_columns = loaded_artifact.get('feature_columns') # We re-run prep, so don't strictly need these here
                    # target_columns = loaded_artifact.get('target_columns')
                    if x_scaler is None or y_scaler is None:
                         print(f"Warning: Loaded artifact for Feeder={feeder_id}, Arch={architecture}, Scenario={scenario} is missing scalers. Skipping.")
                         continue

                    # 2. Prepare Data using Loaded Scalers
                    print("Preparing training and validation data using loaded scalers...")
                    change_in_load = "Change_in_Load" in architecture
                    # Prepare Training Data
                    X_train_scaled, y_train_scaled, _, _ = feature_engineer_and_scale(
                        train_df_raw, scenario, x_scaler=x_scaler, y_scaler=y_scaler, change_in_load=change_in_load, apply_scaling=True
                    )
                    # Prepare Validation Data
                    X_val_scaled, y_val_scaled, _, _ = feature_engineer_and_scale(
                        val_df_raw, scenario, x_scaler=x_scaler, y_scaler=y_scaler, change_in_load=change_in_load, apply_scaling=True
                    )

                    if X_train_scaled.empty or y_train_scaled.empty or X_val_scaled.empty or y_val_scaled.empty:
                         print("Warning: Data preparation resulted in empty dataframes after applying loaded scalers. Skipping.")
                         continue

                    # 3. Define and Run Optuna Study
                    study_name = f"tune-{feeder_id}-{architecture}-{scenario}"
                    study = optuna.create_study(direction=OPTUNA_DIRECTION, study_name=study_name, load_if_exists=False) # Don't resume previous studies

                    # Use partial to pass fixed arguments to the objective function
                    objective_partial = partial(
                        objective,
                        architecture=architecture,
                        X_train_scaled=X_train_scaled,
                        y_train_scaled=y_train_scaled,
                        X_val_scaled=X_val_scaled,
                        y_val_scaled=y_val_scaled
                    )

                    print(f"Starting Optuna optimization ({n_trials} trials)...")
                    study.optimize(objective_partial, n_trials=n_trials, timeout=None) # No timeout

                    # 4. Store and Print Best Results
                    best_params = study.best_params
                    best_value = study.best_value
                    result_key = (feeder_id, architecture, scenario)
                    best_params_all[result_key] = {'best_params': best_params, 'best_value': best_value}

                    print(f"\n--- Best Results for Feeder={feeder_id}, Arch={architecture}, Scenario={scenario} ---")
                    print(f"Best Validation Score ({OPTUNA_METRIC}, scaled): {best_value:.6f}")
                    print("Best Hyperparameters:")
                    for param, value in best_params.items():
                        print(f"  {param}: {value}")

                except FileNotFoundError as e:
                     print(f"ERROR: Could not load required artifact for {architecture}/{scenario}: {e}. Skipping.")
                     continue
                except Exception as e:
                    print(f"!!! ERROR during tuning for Feeder={feeder_id}, Arch={architecture}, Scenario={scenario} !!!")
                    print(f"Error message: {e}")
                    traceback.print_exc()
                    continue # Continue to next combination

    print("\n===== Hyperparameter Tuning Summary =====")
    if best_params_all:
        for (f_id, arch, scen), result in best_params_all.items():
            print(f"\nFeeder: {f_id}, Architecture: {arch}, Scenario: {scen}")
            print(f"  Best Score ({OPTUNA_METRIC}, scaled): {result['best_value']:.6f}")
            print(f"  Best Params: {result['best_params']}")
    else:
        print("No successful tuning runs completed.")

    print("\n--- Hyperparameter Tuning Script Finished ---")