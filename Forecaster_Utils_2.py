import os
from datetime import datetime, timedelta
import sys
import traceback
import pandas as pd
import numpy as np
from padasip.filters import FilterRLS
from DB_Manager import DatabaseManager
from Trainer_Utils_2 import feature_engineer
from Scaler_Manager import ScalerManager
import plotly.express as px
from Scaler import Scaler
from ForecasterTool import ForecasterTool

RLS_BUCKET = "rls-combiners"
FORECAST_CSV_PATH = "forecast_output_feeder_{feeder_id}_{scenario}_{type}.csv"


# def initialize_rls_filters(num_outputs):
#     return [FilterRLS(n=2, mu=0.99, w="random") for _ in range(num_outputs)]


# def convert_change_in_load_to_base_load(X, y_pred_change):
#     prev_day_cols = [col for col in X.columns if col.startswith("Prev_Day_Net_Load_Demand_Hour_")]

#     if len(prev_day_cols) != y_pred_change.shape[-1]:
#         print(f"Expected {len(prev_day_cols)} columns, but got {y_pred_change.shape[1]} columns in y_pred_change.")
#         raise ValueError("Mismatch in number of columns between prev day load and change predictions")
#     prev_day_indices = [X.columns.get_loc(col) for col in prev_day_cols]
#     prev_day_load = X.iloc[:, prev_day_indices].values
#     return prev_day_load + y_pred_change


# def predict_with_padasip_rls(rls_filters, actuals, predictions1, predictions2):
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
#     return combined_predictions


# def forecast_range(
#     feeder_id,
#     scenario,
#     forecast_start_date,
#     forecast_end_date,
#     train_start_date,
#     train_end_date,
#     version="latest",
#     model_arch="LSTM",
#     save_daily_rls=True,
#     new_rls=True,
#     tag="main",
# ):
#     if tag != "main":
#         print(f"Tag '{tag}' is not 'main'. Running training and forecasts in experiment mode.")

#     db = DatabaseManager(tag=tag)
#     current_date = pd.to_datetime(forecast_start_date).date()
#     end_date = pd.to_datetime(forecast_end_date).date()

#     all_forecasts_rls = []
#     all_forecasts_base = []

#     while current_date <= end_date:
#         print(f"\nProcessing {scenario} forecast for {current_date} (Feeder {feeder_id})")
#         target_date = current_date + timedelta(days=1)

#         try:
#             df = db.fetch_data(feeder_id, current_date.isoformat(), (current_date + timedelta(days=1)).isoformat())
#             if df.empty:
#                 print(f"No data for {current_date}. Skipping.")
#                 current_date += timedelta(days=1)
#                 continue

#             scaler = Scaler(feeder_id=feeder_id, scenario_type=scenario, train_start_date=train_start_date, train_end_date=train_end_date)
#             df_scaled = scaler.transform(df)

#             # Engineer features and targets using trainer pipeline
#             X, y_base, y_change, common_timestamps = feature_engineer(df_scaled, feeder_id, scenario, version)

#             if X.empty or y_base.empty or y_change.empty:
#                 print(f"Skipping {current_date}: Feature engineering returned empty results.")
#                 current_date += timedelta(days=1)
#                 continue

#             base_model, change_model = db.load_models_from_supabase(feeder_id, scenario, arch_type=model_arch)

#             if model_arch == "LSTM":
#                 X_numpy = X.values.reshape((1, 1, X.shape[1]))
#             elif model_arch == "ANN" or model_arch == "LightGBM":
#                 X_numpy = X.values.reshape((1, X.shape[1]))

#             y_pred_base = base_model.predict(X_numpy)

#             y_pred_base_true = pd.DataFrame(y_pred_base.flatten(), columns=["Net_Load_Demand"], index=common_timestamps)
#             y_pred_base_true = scaler.inverse_transform(y_pred_base_true)
#             actuals = pd.DataFrame(y_base.values.flatten(), columns=["Net_Load_Demand"], index=common_timestamps)
#             actuals = scaler.inverse_transform(actuals)

#             y_pred_base_true = y_pred_base_true.rename(columns={"Net_Load_Demand": "forecast_value"})
#             actuals = actuals.rename(columns={"Net_Load_Demand": "actual_value"})

#             if change_model is not None:
#                 y_pred_change = change_model.predict(X_numpy) if change_model else np.zeros_like(y_pred_base)
#                 y_pred_combined_change_prev_day_base = convert_change_in_load_to_base_load(X, y_pred_change)
#                 print("Converted load shape: ", y_pred_combined_change_prev_day_base.shape)

#                 # Load or initialize RLS filters
#                 rls_filters = db.load_rls_filters(feeder_id, version, scenario, model_arch)
#                 if rls_filters is None or new_rls:
#                     print("Initializing new RLS filters.")
#                     new_rls = False
#                     rls_filters = initialize_rls_filters(y_pred_base.shape[-1])

#                 y_pred_rls_scaled = predict_with_padasip_rls(rls_filters, y_base.values, y_pred_base, y_pred_combined_change_prev_day_base)
#                 print("RLS prediction: ", y_pred_rls_scaled)
#                 print("True: ", y_base)
#                 print("Base prediction: ", y_pred_base)
#                 print("Change converted prediction: ", y_pred_combined_change_prev_day_base)
#                 y_pred_rls_scaled = pd.DataFrame(y_pred_rls_scaled.flatten(), columns=["Net_Load_Demand"], index=common_timestamps)

#                 y_pred_rls = scaler.inverse_transform(y_pred_rls_scaled)
#                 y_pred_rls = y_pred_rls.rename(columns={"Net_Load_Demand": "forecast_value"})
#                 y_pred_rls = pd.concat([y_pred_rls, actuals], axis=1)
#                 all_forecasts_rls.append(y_pred_rls)

#                 if save_daily_rls:
#                     db.save_rls_filters(rls_filters, feeder_id, version, scenario, model_arch)

#                 db.save_forecasts(
#                     feeder_id,
#                     version,
#                     scenario_type=scenario,
#                     model_architecture_type=model_arch,
#                     forecasts_df=y_pred_rls,
#                 )
#             else:
#                 db.save_forecasts(
#                     feeder_id,
#                     version,
#                     scenario_type=scenario,
#                     model_architecture_type=model_arch,
#                     forecasts_df=y_pred_base_true,
#                 )

#             all_forecasts_base.append(y_pred_base_true)

#         except Exception as e:
#             print(f"Error on {current_date}: {e}")
#             traceback.print_exc()

#         current_date += timedelta(days=1)

#     # Save to CSV
#     if all_forecasts_rls:
#         result_df = pd.concat(all_forecasts_rls)
#         result_df.to_csv(FORECAST_CSV_PATH.format(feeder_id=feeder_id, scenario=scenario, type="rls"))
#         print(f"Forecast results saved to {FORECAST_CSV_PATH.format(feeder_id=feeder_id, scenario=scenario, type='rls')}")

#     if all_forecasts_base:
#         result_df = pd.concat(all_forecasts_base)
#         result_df.to_csv(FORECAST_CSV_PATH.format(feeder_id=feeder_id, scenario=scenario, type="base"))
#         print(f"Forecast results saved to {FORECAST_CSV_PATH.format(feeder_id=feeder_id, scenario=scenario, type='base')}")
#     else:
#         print("No forecasts generated.")


if __name__ == "__main__":
    for i in range(1, 2):
        print(f"Running forecast for feeder {i}...")
        for model_arch in ["ANN"]:
            forecast_range(
                feeder_id=i,
                scenario="24hr",
                forecast_start_date="2024-05-01",
                forecast_end_date="2024-07-30",
                train_start_date="2024-01-01",
                train_end_date="2024-05-31",
                version="v1.6_Fresh_Testing_6",
                new_rls=True,
                model_arch=model_arch,
                tag=f"exp_{model_arch}" if model_arch == "LightGBM" or model_arch == "ANN" else "main",
            )
