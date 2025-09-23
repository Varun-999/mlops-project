FROM python:3.11-slim

WORKDIR /app

# # Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create artifacts directory (empty for now)
RUN mkdir -p /app/artifacts

# Copy source code
COPY src/ /app/src/

# Set environment variables
ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]



# # 1. Start with an official Python base image
# FROM python:3.11-slim

# # 2. Set the working directory inside the container
# WORKDIR /app

# # 3. Copy only the requirements file first
# COPY requirements.txt .

# # 4. Install Python dependencies with --no-cache-dir
# RUN pip install --no-cache-dir -r requirements.txt

# # 5. Copy only the essential artifacts
# COPY ./artifacts/label_encoder.joblib /app/artifacts/
# COPY ./artifacts/normalization_stats.joblib /app/artifacts/
# COPY ./artifacts/model.keras /app/artifacts/

# # 6. Copy only the necessary application code
# COPY src/main.py .
# COPY src/config.py .
# COPY src/data_processing.py .

# ENV PYTHONPATH=/app/src:/app
# ENV PYTHONUNBUFFERED=1

# # 7. Expose the port the app runs on
# EXPOSE 8000

# # 8. Define the command to run your app
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



# # Use Python 3.11 slim image
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies for audio processing
# RUN apt-get update && apt-get install -y \
#     libsndfile1 \
#     ffmpeg \
#     libsndfile1-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements file
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Create artifacts directory
# RUN mkdir -p /app/artifacts

# # Copy artifacts
# COPY ./artifacts/label_encoder.joblib /app/artifacts/
# COPY ./artifacts/normalization_stats.joblib /app/artifacts/
# COPY ./artifacts/model.keras /app/artifacts/

# # Copy application code
# COPY src/main.py .
# COPY src/config.py .
# COPY src/data_processing.py .

# # Expose port (Render will set PORT environment variable)
# EXPOSE 8000

# # Command to run the app with dynamic port binding
# CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}



# Use Python 3.11 slim image
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app


# # Install system dependencies for audio processing
# RUN apt-get update && apt-get install -y \
#     libsndfile1 \
#     ffmpeg \
#     libsndfile1-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements file
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Create artifacts directory
# RUN mkdir -p /app/artifacts

# # Copy artifacts
# COPY ./artifacts/label_encoder.joblib /app/artifacts/
# COPY ./artifacts/normalization_stats.joblib /app/artifacts/
# COPY ./artifacts/model.keras /app/artifacts/

# # # Copy all source files
# # COPY src/ /app/src/

# # Set Python path to include src directory
# ENV PYTHONPATH=/app/src:/app

# # Expose port
# EXPOSE 8000

# # Command to run the app - using the file from src directory
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]