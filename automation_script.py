import subprocess
from datetime import datetime

with open("output_log.txt", "w") as f:
    # Run the first script
    f.write(f"\n--- Running Train_All_Feeders.py at {datetime.now()} ---\n")
    subprocess.run(["python", "Train_All_Feeders.py"], stdout=f, stderr=subprocess.STDOUT)

    # Run the second script after the first one finishes
    f.write(f"\n--- Running run_forecasts.py at {datetime.now()} ---\n")
    subprocess.run(["python", "run_forecasts.py"], stdout=f, stderr=subprocess.STDOUT)
