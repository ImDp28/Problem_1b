# Dockerfile
# Stage 1: Base image with Python
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and run the model download script to bake models into the image
COPY download_models.py .
RUN python download_models.py

# Copy the application code into the container
COPY main.py .
COPY document_analyst.py .

# Define the entry point for the container
ENTRYPOINT ["python", "main.py"]