# Use miniconda as base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment file and all project files
COPY environment.yml .
COPY . .

RUN conda update -n base -c defaults conda

# Create the conda environment
RUN conda env create -f environment.yml

# Activate environment and set it as default
SHELL ["conda", "run", "-n", "forecast-env", "/bin/bash", "-c"]

# Ensure the shell script is executable
RUN chmod +x Automated_Forecast_Runner.sh

# Default command (can be overridden)
CMD ["conda", "run", "--no-capture-output", "-n", "forecast-env", "bash", "./Automated_Forecast_Runner.sh"]
