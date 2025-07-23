# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set up environment to be non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install the default python3, venv, and pip provided by the base image
RUN apt-get update && \
    apt-get install -y python3 python3-venv python3-pip && \
    apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code and model artifacts
COPY ./app ./app
COPY ./model ./model

# Make port 8501 available
EXPOSE 8501

# Run main.py when the container launches
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.enableCORS=false"]