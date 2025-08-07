# 1. Start with an official Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the Python dependencies
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code into the container
# This includes main.py, config.py, data_processing.py, etc.
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Define the command to run your app when the container starts
# We use 0.0.0.0 to make it accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
