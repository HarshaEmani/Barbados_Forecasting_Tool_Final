# run_forecasts.py
import subprocess
from datetime import datetime, timedelta
import time

start_date = datetime.strptime("2024-07-26", "%Y-%m-%d")
end_date = datetime.strptime("2024-07-26", "%Y-%m-%d")

current_date = start_date

while current_date <= end_date:
    target_date_str = current_date.strftime("%Y-%m-%d")
    print(f"Running forecast for {target_date_str}")

    subprocess.run(["python", "Forecast_All_Feeders.py", target_date_str])

    current_date += timedelta(days=1)

    # if current_date <= end_date:
    #     print("Waiting 10 minutes before next run...")
    #     time.sleep(600)  # wait 10 minutes
