# Use an official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker cache
COPY requirements.txt .

# Create and activate virtual environment
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Make venv binaries accessible
ENV PATH="/opt/venv/bin:$PATH"

# Copy the rest of the project
COPY . .

# Ensure the shell script is executable
RUN chmod +x Automated_Forecast_Runner.sh

# Default command
CMD ["bash", "./Automated_Forecast_Runner.sh"]



# # Use micromamba as the base image
# FROM mambaorg/micromamba:1.5.7

# # Set up micromamba environment activation
# ENV MAMBA_DOCKERFILE_ACTIVATE=1

# # Set working directory
# WORKDIR /app

# # Copy environment.yml first to leverage Docker cache
# COPY environment.yml .

# # Create the conda environment using micromamba
# RUN micromamba create -n forecast-env -f environment.yml && \
#     micromamba clean --all --yes

# # Copy the rest of the project files
# COPY . .

# # Ensure the shell script is executable
# RUN chmod +x Automated_Forecast_Runner.sh

# # Use micromamba shell to run commands inside the env
# SHELL ["micromamba", "run", "-n", "forecast-env", "/bin/bash", "-c"]

# # Default command
# CMD ["micromamba", "run", "-n", "forecast-env", "bash", "./Automated_Forecast_Runner.sh"]




# # Use miniconda as base image
# FROM continuumio/miniconda3

# # Set working directory
# WORKDIR /app

# # Copy environment file and all project files
# COPY environment.yml .
# COPY . .

# RUN conda update -n base -c defaults conda

# # Create the conda environment
# # RUN conda env create -f environment.yml

# RUN conda install -n base -c conda-forge mamba
# RUN mamba env create -f environment.yml

# # Activate environment and set it as default
# SHELL ["conda", "run", "-n", "forecast-env", "/bin/bash", "-c"]

# # Ensure the shell script is executable
# RUN chmod +x Automated_Forecast_Runner.sh

# # Default command (can be overridden)
# CMD ["conda", "run", "--no-capture-output", "-n", "forecast-env", "bash", "./Automated_Forecast_Runner.sh"]
