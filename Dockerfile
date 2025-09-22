# 1. Start with an official Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first
COPY requirements.txt .

# 4. Install Python dependencies with --no-cache-dir
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy only the essential artifacts
COPY ./artifacts/label_encoder.joblib /app/artifacts/
COPY ./artifacts/normalization_stats.joblib /app/artifacts/
COPY ./artifacts/model.keras /app/artifacts/

# 6. Copy only the necessary application code
COPY src/main.py .
COPY src/config.py .
COPY src/data_processing.py .

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Define the command to run your app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
