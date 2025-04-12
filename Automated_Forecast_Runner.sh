#!/bin/bash

cd /app  # inside container working dir

# File path inside the mounted volume
LAST_DATE_FILE="data/Last_Forecast_Date.txt"

# Initialize with first test date if file doesn't exist
if [ ! -f "$LAST_DATE_FILE" ]; then
    echo "2025-07-01" > "$LAST_DATE_FILE"
    echo "Initialized with first test date"
fi

# Read and compute next date
LAST_DATE=$(cat "$LAST_DATE_FILE")
NEXT_DATE=$(date -I -d "$LAST_DATE + 1 day")

# Stop if date exceeds test range
if [ "$NEXT_DATE" \> "2025-07-03" ]; then
    echo "All forecasts completed."
    exit 0
fi

# Run the Python forecast script with --target_date
conda run -n forecast-env python3 forecast.py --target_date "$NEXT_DATE"

# Update the date file
echo "$NEXT_DATE" > "$LAST_DATE_FILE"


# #!/bin/bash

# cd /path/to/project

# # File that stores the last forecasted date
# LAST_DATE_FILE="Last_Forecast_Date.txt"

# # If the file doesn't exist, initialize with the first test date
# if [ ! -f "$LAST_DATE_FILE" ]; then
#     echo "2025-07-01" > "$LAST_DATE_FILE"  # Example: first day of test period
# fi

# # Read the last forecasted date
# LAST_DATE=$(cat "$LAST_DATE_FILE")
# NEXT_DATE=$(date -I -d "$LAST_DATE + 1 day")

# # Stop if next date exceeds test data
# if [ "$NEXT_DATE" \> "2025-07-03" ]; then
#     echo "All forecasts completed."
#     exit 0
# fi

# # Call forecast.py with the next date
# /usr/bin/python3 forecast.py --target_date "$NEXT_DATE"

# # Save the new last forecasted date
# echo "$NEXT_DATE" > "$LAST_DATE_FILE"