#v3
import os
import io
import tensorflow
import joblib
import numpy as np
import librosa # You'll need this for audio processing
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
# --- CHANGE 1: Import the Keras load_model function ---
import tensorflow as tf

# --- CHANGE 2: MLflow imports are no longer needed ---
# from mlflow.tracking import MlflowClient
# from mlflow.exceptions import MlflowException
# import mlflow

# --- Custom Module Imports ---
# This remains the same
import config
import data_processing

# --- Configuration ---
# These are no longer needed as we load directly from a local folder
# REGISTERED_MODEL_NAME = "AudioClassifier"
# MODEL_ALIAS = "Production"

# --- 1. Define Application and Global Variables ---
app = FastAPI(title="Audio Classification API")

ml_model = None
label_encoder = None
norm_stats = None

# --- 2. Define Request/Response Models ---
# This remains the same
class PredictionResponse(BaseModel):
    predicted_class: str

# --- 3. Implement a Startup Event to Load the Model ---
# This entire function is replaced to load artifacts locally.
@app.on_event("startup")
def load_artifacts():
    """
    Loads the production model and preprocessing artifacts from the local '/app/artifacts'
    directory inside the container.
    """
    global tf, ml_model, label_encoder, norm_stats
    print("Loading model and artifacts from local directory...")

    try:
        # Define the paths to the artifacts inside the container
        # These paths correspond to the `COPY ./artifacts /app/artifacts` command in your Dockerfile
        artifacts_dir = "./artifacts"
        # model_path = os.path.join(artifacts_dir, "model","model.keras") # Path to the saved model folder
        model_path = os.path.join(artifacts_dir,"model.keras")
        le_path = os.path.join(artifacts_dir, "label_encoder.joblib")
        stats_path = os.path.join(artifacts_dir, "normalization_stats.joblib")

        # Load the artifacts
        # ml_model = load_model(model_path)
        # Load the artifacts
        ml_model = tf.keras.models.load_model(model_path)   # keras model
        label_encoder = joblib.load(le_path)
        norm_stats = joblib.load(stats_path)

        print("✅ Model and artifacts loaded successfully.")

    except Exception as e:
        print(f"❌ An error occurred during artifact loading: {e}")
        # This is a critical error, so we raise it to stop the server from starting improperly
        raise RuntimeError(f"Could not load model or artifacts: {e}")


# --- 4. Define the Prediction Endpoint ---
# The logic inside this function remains almost identical.
@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts an audio file, preprocesses it, and returns the predicted class.
    """
    if not ml_model or not label_encoder or not norm_stats:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")

    # Read the uploaded audio file in memory
    audio_bytes = await file.read()

    try:
        # --- Preprocess the new audio data ---
        # 1. Load audio data from in-memory bytes
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=config.SAMPLE_RATE, duration=config.DURATION)
        
        # 2. Extract MFCCs
        mfccs = data_processing.extract_mfcc(audio)
        
        # 3. Normalize features using the loaded stats
        mfccs_normalized = (mfccs - norm_stats['mean']) / (norm_stats['std'] + 1e-8)
        
        # 4. Reshape for the model's input (batch, time_steps, n_mfcc, channels)
        mfccs_reshaped = mfccs_normalized[np.newaxis, ..., np.newaxis]

        # --- Make Prediction ---
        # Use the predict method of the loaded Keras model
        prediction_vector = ml_model.predict(mfccs_reshaped)
        
        # --- Post-process the Prediction ---
        predicted_index = np.argmax(prediction_vector, axis=1)[0]
        predicted_class = label_encoder.inverse_transform([predicted_index])[0]

        return {"predicted_class": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file or make prediction: {e}")









# # v2 working locally
# import os
# import io
# import joblib
# import numpy as np
# import pandas as pd
# import mlflow
# import librosa # You'll need this for audio processing
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel
# from mlflow.tracking import MlflowClient
# from mlflow.exceptions import MlflowException

# # --- Custom Module Imports ---
# # NOTE: You must have your config.py and data_processing.py files
# # available in the same directory or in your PYTHONPATH for this to work.
# import config
# import data_processing

# # --- Configuration ---
# # This should match the name you used in your training script
# REGISTERED_MODEL_NAME = "AudioClassifier"
# # This should match the alias you set for your best model
# MODEL_ALIAS = "Production"

# # --- 1. Define Application and Global Variables ---
# # Create the FastAPI app instance
# app = FastAPI(title="Audio Classification API")

# # These global variables will hold the loaded model and preprocessing objects.
# # They are initialized to None and will be loaded by the startup event.
# ml_model = None
# label_encoder = None
# norm_stats = None

# # --- 2. Define Request/Response Models (Good Practice) ---
# class PredictionResponse(BaseModel):
#     predicted_class: str
    
# # --- 3. Implement a Startup Event to Load the Model ---
# # This function will be executed once when the FastAPI server starts.
# # It's the perfect place to load heavy objects like the ML model.
# @app.on_event("startup")
# def load_production_model():
#     """
#     Loads the production model and preprocessing artifacts from MLflow by resolving the alias.
#     """
#     global ml_model, label_encoder, norm_stats
#     print(f"Loading model '{REGISTERED_MODEL_NAME}' with alias '{MODEL_ALIAS}'...")
#     client = MlflowClient()

#     try:
#         # --- THIS IS THE FIX ---
#         # 1. Get the version number associated with the 'Production' alias
#         model_version_details = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
#         version_number = model_version_details.version
#         run_id = model_version_details.run_id
#         print(f"Found version {version_number} for alias '{MODEL_ALIAS}'.")

#         # 2. Construct the model URI using the specific version number
#         model_uri = f"models:/{REGISTERED_MODEL_NAME}/{version_number}"
        
#         # 3. Load the model from the registry using the version-specific URI
#         ml_model = mlflow.pyfunc.load_model(model_uri)
#         print("✅ Model loaded successfully.")

#         # --- Load Preprocessing Artifacts ---
#         # Create a temporary directory to download artifacts
#         local_dir = "/FtoL/"
#         if not os.path.exists(local_dir):
#             os.mkdir(local_dir)

#         # Download the specific preprocessing artifacts from the correct run
#         print(f"Downloading preprocessing artifacts from run_id: {run_id}")
#         client.download_artifacts(run_id, "preprocessing", local_dir)
        
#         # Load the artifacts using joblib
#         le_path = os.path.join(local_dir, "preprocessing", config.LABEL_ENCODER_ARTIFACT_NAME)
#         stats_path = os.path.join(local_dir, "preprocessing", config.NORMALIZATION_STATS_ARTIFACT_NAME)
        
#         label_encoder = joblib.load(le_path)
#         norm_stats = joblib.load(stats_path)
#         print("✅ Preprocessing artifacts loaded successfully.")

#     except MlflowException as e:
#         # This error is expected if the alias does not exist on the first run
#         if "RESOURCE_DOES_NOT_EXIST" in str(e):
#             print(f"❌ Could not find model '{REGISTERED_MODEL_NAME}' with alias '{MODEL_ALIAS}'.")
#             raise RuntimeError(f"Model with alias '{MODEL_ALIAS}' not found.")
#         else:
#             raise e
#     except Exception as e:
#         print(f"❌ An unexpected error occurred: {e}")
#         raise RuntimeError(f"Could not load model or artifacts: {e}")


# # --- 4. Define the Prediction Endpoint ---
# @app.post("/predict/", response_model=PredictionResponse)
# async def predict(file: UploadFile = File(...)):
#     """
#     Accepts an audio file, preprocesses it, and returns the predicted class.
#     """
#     if not ml_model or not label_encoder or not norm_stats:
#         raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")

#     # Read the uploaded audio file in memory
#     audio_bytes = await file.read()

#     try:
#         # --- Preprocess the new audio data ---
#         # This logic should EXACTLY match the preprocessing from your training script.
        
#         # 1. Load audio data (using a BytesIO object to read from memory)
#         audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
#         # 2. Extract MFCCs (using your data_processing functions)
#         mfccs = data_processing.extract_mfcc(audio)
        
#         # 3. Normalize features using the loaded stats
#         mfccs_normalized = (mfccs - norm_stats['mean']) / (norm_stats['std'] + 1e-8)
        
#         # 4. Reshape for the model's input
#         mfccs_reshaped = mfccs_normalized[np.newaxis, ..., np.newaxis]

#         # --- Make Prediction ---
#         # The pyfunc wrapper for a TensorFlow model logged with a numpy input_example
#         # can often handle a numpy array directly. If you encounter issues,
#         # wrapping it in a DataFrame is a common alternative.
#         prediction_result = ml_model.predict(mfccs_reshaped)

#         # The result might be a nested array, so we flatten and get the probabilities
#         probabilities = np.array(prediction_result).flatten()
        
#         # --- Post-process the Prediction ---
#         # Get the index of the highest probability
#         predicted_index = np.argmax(probabilities)
        
#         # Convert the index back to the original string label
#         predicted_class = label_encoder.inverse_transform([predicted_index])[0]

#         return {"predicted_class": predicted_class}

#     except Exception as e:
#         # Return an error if anything goes wrong during processing or prediction
#         raise HTTPException(status_code=500, detail=f"Failed to process file or make prediction: {e}")












#v1
# import os
# import io
# import joblib
# import numpy as np
# import mlflow
# import librosa # You'll need this for audio processing
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel

# # --- Configuration ---
# # This should match the name you used in your training script
# REGISTERED_MODEL_NAME = "AudioClassifier"
# # This should match the alias you set for your best model
# MODEL_ALIAS = "Production"

# # --- 1. Define Application and Global Variables ---
# # Create the FastAPI app instance
# app = FastAPI(title="Audio Classification API")

# # These global variables will hold the loaded model and preprocessing objects.
# # They are initialized to None and will be loaded by the startup event.
# ml_model = None
# label_encoder = None
# norm_stats = None

# # --- 2. Define Request/Response Models (Good Practice) ---
# class PredictionResponse(BaseModel):
#     predicted_class: str
    
# # --- 3. Implement a Startup Event to Load the Model ---
# # This function will be executed once when the FastAPI server starts.
# # It's the perfect place to load heavy objects like the ML model.
# @app.on_event("startup")
# def load_production_model():
#     """
#     Loads the production model and preprocessing artifacts from MLflow.
#     """
#     global ml_model, label_encoder, norm_stats
#     print(f"Loading model '{REGISTERED_MODEL_NAME}' with alias '{MODEL_ALIAS}'...")

#     try:
#         # Load the model from the registry using its alias
#         # mlflow.pyfunc.load_model is the most generic way and is recommended.
#         model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_ALIAS}"
#         ml_model = mlflow.pyfunc.load_model(model_uri)
#         print("✅ Model loaded successfully.")

#         # --- Load Preprocessing Artifacts ---
#         # We need to get the run_id from the model version to download its artifacts
#         client = mlflow.tracking.MlflowClient()
#         model_version_details = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
#         run_id = model_version_details.run_id
        
#         # Create a temporary directory to download artifacts
#         local_dir = "/tmp/mlflow_artifacts"
#         if not os.path.exists(local_dir):
#             os.mkdir(local_dir)

#         # Download the specific preprocessing artifacts
#         print(f"Downloading preprocessing artifacts from run_id: {run_id}")
#         client.download_artifacts(run_id, "preprocessing", local_dir)
        
#         # Load the artifacts using joblib
#         le_path = os.path.join(local_dir, "preprocessing", config.LABEL_ENCODER_ARTIFACT_NAME)
#         stats_path = os.path.join(local_dir, "preprocessing", config.NORMALIZATION_STATS_ARTIFACT_NAME)
        
#         label_encoder = joblib.load(le_path)
#         norm_stats = joblib.load(stats_path)
#         print("✅ Preprocessing artifacts loaded successfully.")

#     except Exception as e:
#         print(f"❌ Error loading model or artifacts: {e}")
#         # In a real application, you might want to prevent the server
#         # from starting if the model can't be loaded.
#         raise RuntimeError(f"Could not load model or artifacts: {e}")


# # --- 4. Define the Prediction Endpoint ---
# @app.post("/predict/", response_model=PredictionResponse)
# async def predict(file: UploadFile = File(...)):
#     """
#     Accepts an audio file, preprocesses it, and returns the predicted class.
#     """
#     if not ml_model or not label_encoder or not norm_stats:
#         raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")

#     # Read the uploaded audio file in memory
#     audio_bytes = await file.read()

#     try:
#         # --- Preprocess the new audio data ---
#         # This logic should EXACTLY match the preprocessing from your training script.
        
#         # 1. Load audio data (using a BytesIO object to read from memory)
#         audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
#         # 2. Extract MFCCs (using your data_processing functions)
#         # NOTE: You would import your 'data_processing' module here
#         mfccs = data_processing.extract_mfcc(audio, sample_rate)
        
#         # 3. Normalize features using the loaded stats
#         mfccs_normalized = (mfccs - norm_stats['mean']) / (norm_stats['std'] + 1e-8)
        
#         # 4. Reshape for the model's input
#         mfccs_reshaped = mfccs_normalized[np.newaxis, ..., np.newaxis]

#         # --- Make Prediction ---
#         # The pyfunc model expects a pandas DataFrame, so we wrap the input.
#         # The column names don't matter here as the model expects a single tensor input.
#         prediction_input = pd.DataFrame([mfccs_reshaped])
#         prediction_result = ml_model.predict(prediction_input)

#         # The result might be a nested array, so we flatten and get the probabilities
#         probabilities = np.array(prediction_result).flatten()
        
#         # --- Post-process the Prediction ---
#         # Get the index of the highest probability
#         predicted_index = np.argmax(probabilities)
        
#         # Convert the index back to the original string label
#         predicted_class = label_encoder.inverse_transform([predicted_index])[0]

#         return {"predicted_class": predicted_class}

#     except Exception as e:
#         # Return an error if anything goes wrong during processing or prediction
#         raise HTTPException(status_code=500, detail=f"Failed to process file or make prediction: {e}")

