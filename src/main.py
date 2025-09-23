import os
import io
import tensorflow as tf
import joblib
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Try to import custom modules - handle gracefully if missing
try:
    import config
    SAMPLE_RATE = config.SAMPLE_RATE
    DURATION = config.DURATION
except ImportError:
    SAMPLE_RATE = 22050
    DURATION = 4
    logging.warning("Config module not found, using default values")

try:
    import data_processing
except ImportError:
    data_processing = None
    logging.warning("Data processing module not found, will use fallback methods")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Urban Sound Classification API",
    description="Audio classification API for Urban Sound dataset",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
ml_model = None
label_encoder = None
norm_stats = None
model_loaded = False

# Response models
class PredictionResponse(BaseModel):
    predicted_class: str

class HealthResponse(BaseModel):
    status: str
    message: str

@app.on_event("startup")
async def load_artifacts():
    """Load model artifacts on startup if available"""
    global ml_model, label_encoder, norm_stats, model_loaded
    
    logger.info("Starting artifact loading process...")
    
    try:
        # Define artifact paths
        artifacts_dir = "/app/artifacts"
        model_path = os.path.join(artifacts_dir, "model.keras")
        le_path = os.path.join(artifacts_dir, "label_encoder.joblib")
        stats_path = os.path.join(artifacts_dir, "normalization_stats.joblib")
        
        logger.info(f"Checking artifacts directory: {artifacts_dir}")
        logger.info(f"Directory exists: {os.path.exists(artifacts_dir)}")
        
        if os.path.exists(artifacts_dir):
            logger.info(f"Contents of artifacts directory: {os.listdir(artifacts_dir)}")
        
        # Check if all required files exist
        missing_files = []
        for path, name in [(model_path, "model.keras"), (le_path, "label_encoder.joblib"), (stats_path, "normalization_stats.joblib")]:
            if not os.path.exists(path):
                missing_files.append(name)
                logger.info(f"Missing file: {name} at {path}")
            else:
                logger.info(f"Found file: {name} at {path}")
        
        if missing_files:
            logger.warning(f"Missing artifact files: {', '.join(missing_files)}")
            logger.info("API will run in demo mode without prediction capability")
            model_loaded = False
            return
        
        # Try to load artifacts
        logger.info("All artifact files found, attempting to load...")
        
        logger.info("Loading TensorFlow model...")
        ml_model = tf.keras.models.load_model(model_path)
        
        logger.info("Loading label encoder...")
        label_encoder = joblib.load(le_path)
        
        logger.info("Loading normalization stats...")
        norm_stats = joblib.load(stats_path)
        
        model_loaded = True
        logger.info("SUCCESS: Model and artifacts loaded successfully!")
        logger.info(f"Model input shape: {ml_model.input_shape}")
        logger.info(f"Available classes: {len(label_encoder.classes_)}")
        
    except Exception as e:
        logger.error(f"Error during artifact loading: {str(e)}")
        logger.info("API will run in demo mode without prediction capability")
        model_loaded = False
        # Don't raise the error - let the app start in demo mode

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    if model_loaded:
        return {
            "status": "ready", 
            "message": "Urban Sound Classification API is running with full prediction capability! Visit /docs for API documentation."
        }
    else:
        return {
            "status": "demo", 
            "message": "Urban Sound Classification API is running in demo mode (no model loaded). Visit /docs for API documentation."
        }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    if model_loaded:
        return {
            "status": "healthy",
            "message": "All systems operational - ready for predictions"
        }
    else:
        return {
            "status": "healthy", 
            "message": "API running in demo mode - service is healthy but no model loaded"
        }

@app.get("/info")
async def get_model_info():
    """Get information about the loaded model and current status"""
    base_info = {
        "tensorflow_version": tf.__version__,
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "model_loaded": model_loaded
    }
    
    if model_loaded:
        return {
            **base_info,
            "status": "ready",
            "model_input_shape": str(ml_model.input_shape),
            "available_classes": label_encoder.classes_.tolist(),
            "message": "Model loaded and ready for predictions"
        }
    else:
        return {
            **base_info,
            "status": "demo mode",
            "message": "No model loaded. Prediction functionality not available. API running in demo mode.",
            "available_endpoints": ["/", "/health", "/info", "/docs"]
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Audio classification endpoint"""
    
    # Check if model is loaded
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available. Model artifacts not loaded. API is running in demo mode. Please check /info for current status."
        )
    
    # Validate file type
    allowed_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    logger.info(f"Processing prediction for file: {file.filename}")
    
    try:
        # Read uploaded audio file
        audio_bytes = await file.read()
        logger.info(f"File size: {len(audio_bytes)} bytes")
        
        # Load audio data from memory
        audio, _ = librosa.load(
            io.BytesIO(audio_bytes), 
            sr=SAMPLE_RATE, 
            duration=DURATION
        )
        logger.info(f"Audio loaded: {len(audio)} samples at {SAMPLE_RATE}Hz")
        
        # Extract MFCC features
        if data_processing:
            mfccs = data_processing.extract_mfcc(audio)
        else:
            # Fallback MFCC extraction
            mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
            mfccs = mfccs.T  # Transpose to match expected shape
        
        logger.info(f"MFCC features extracted: shape {mfccs.shape}")
        
        # Normalize features
        mfccs_normalized = (mfccs - norm_stats['mean']) / (norm_stats['std'] + 1e-8)
        
        # Reshape for model input (batch, time_steps, n_mfcc, channels)
        mfccs_reshaped = mfccs_normalized[np.newaxis, ..., np.newaxis]
        logger.info(f"Input shape for model: {mfccs_reshaped.shape}")
        
        # Make prediction
        prediction_vector = ml_model.predict(mfccs_reshaped, verbose=0)
        
        # Post-process prediction
        predicted_index = np.argmax(prediction_vector, axis=1)[0]
        predicted_class = label_encoder.inverse_transform([predicted_index])[0]
        confidence = float(np.max(prediction_vector))
        
        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
        
        return {"predicted_class": predicted_class}
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file or make prediction: {str(e)}"
        )









#working with ci file but failing on render
# import os
# import io
# import tensorflow as tf
# import joblib
# import numpy as np
# import librosa
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import logging

# # Import custom modules
# import config
# import data_processing

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # FastAPI app instance
# app = FastAPI(
#     title="Urban Sound Classification API",
#     description="Audio classification API for Urban Sound dataset",
#     version="1.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global variables
# ml_model = None
# label_encoder = None
# norm_stats = None

# # Response models
# class PredictionResponse(BaseModel):
#     predicted_class: str

# class HealthResponse(BaseModel):
#     status: str
#     message: str

# @app.on_event("startup")
# async def load_artifacts():
#     """Load model artifacts on startup"""
#     global ml_model, label_encoder, norm_stats
    
#     logger.info("Loading model and artifacts from local directory...")
    
#     try:
#         # Define artifact paths relative to the working directory
#         artifacts_dir = "/app/artifacts"
#         model_path = os.path.join(artifacts_dir, "model.keras")
#         le_path = os.path.join(artifacts_dir, "label_encoder.joblib")
#         stats_path = os.path.join(artifacts_dir, "normalization_stats.joblib")
        
#         # Log current working directory and check file existence
#         logger.info(f"Current working directory: {os.getcwd()}")
#         logger.info(f"Files in /app: {os.listdir('/app') if os.path.exists('/app') else 'N/A'}")
#         logger.info(f"Files in /app/artifacts: {os.listdir('/app/artifacts') if os.path.exists('/app/artifacts') else 'N/A'}")
        
#         # Verify files exist
#         for path, name in [(model_path, "model"), (le_path, "label encoder"), (stats_path, "normalization stats")]:
#             if not os.path.exists(path):
#                 raise FileNotFoundError(f"{name} file not found at {path}")
#             logger.info(f"{name} found at {path}")
        
#         # Load artifacts
#         logger.info("Loading TensorFlow model...")
#         ml_model = tf.keras.models.load_model(model_path)
        
#         logger.info("Loading label encoder...")
#         label_encoder = joblib.load(le_path)
        
#         logger.info("Loading normalization stats...")
#         norm_stats = joblib.load(stats_path)
        
#         logger.info("Model and artifacts loaded successfully!")
#         logger.info(f"Model input shape: {ml_model.input_shape}")
#         logger.info(f"Available classes: {len(label_encoder.classes_)}")
#         logger.info(f"Classes: {list(label_encoder.classes_)}")
        
#     except Exception as e:
#         logger.error(f"Error loading artifacts: {e}")
#         raise RuntimeError(f"Could not load model or artifacts: {e}")

# @app.get("/", response_model=HealthResponse)
# async def root():
#     """Root endpoint"""
#     return {
#         "status": "success", 
#         "message": "Urban Sound Classification API is running! Visit /docs for API documentation."
#     }

# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """Health check endpoint"""
#     if ml_model is None or label_encoder is None or norm_stats is None:
#         raise HTTPException(
#             status_code=503,
#             detail="Service unavailable - model artifacts not loaded"
#         )
#     return {
#         "status": "healthy",
#         "message": "All systems operational - ready for predictions"
#     }

# @app.get("/info")
# async def get_model_info():
#     """Get information about the loaded model"""
#     if ml_model is None or label_encoder is None:
#         raise HTTPException(
#             status_code=503,
#             detail="Model not loaded"
#         )
    
#     return {
#         "model_input_shape": str(ml_model.input_shape),
#         "available_classes": label_encoder.classes_.tolist(),
#         "sample_rate": config.SAMPLE_RATE,
#         "duration": config.DURATION,
#         "tensorflow_version": tf.__version__
#     }

# @app.post("/predict", response_model=PredictionResponse)
# async def predict(file: UploadFile = File(...)):
#     """Audio classification endpoint"""
    
#     # Check if model is loaded
#     if not ml_model or not label_encoder or not norm_stats:
#         raise HTTPException(
#             status_code=503,
#             detail="Model is not loaded yet. Please wait for initialization."
#         )
    
#     # Validate file type
#     allowed_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')
#     if not file.filename.lower().endswith(allowed_extensions):
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid file format. Supported formats: {', '.join(allowed_extensions)}"
#         )
    
#     logger.info(f"Processing prediction for file: {file.filename}")
    
#     try:
#         # Read uploaded audio file
#         audio_bytes = await file.read()
#         logger.info(f"File size: {len(audio_bytes)} bytes")
        
#         # Load audio data from memory
#         audio, _ = librosa.load(
#             io.BytesIO(audio_bytes), 
#             sr=config.SAMPLE_RATE, 
#             duration=config.DURATION
#         )
#         logger.info(f"Audio loaded: {len(audio)} samples at {config.SAMPLE_RATE}Hz")
        
#         # Extract MFCC features
#         mfccs = data_processing.extract_mfcc(audio)
#         logger.info(f"MFCC features extracted: shape {mfccs.shape}")
        
#         # Normalize features
#         mfccs_normalized = (mfccs - norm_stats['mean']) / (norm_stats['std'] + 1e-8)
        
#         # Reshape for model input (batch, time_steps, n_mfcc, channels)
#         mfccs_reshaped = mfccs_normalized[np.newaxis, ..., np.newaxis]
#         logger.info(f"Input shape for model: {mfccs_reshaped.shape}")
        
#         # Make prediction
#         prediction_vector = ml_model.predict(mfccs_reshaped, verbose=0)
        
#         # Post-process prediction
#         predicted_index = np.argmax(prediction_vector, axis=1)[0]
#         predicted_class = label_encoder.inverse_transform([predicted_index])[0]
#         confidence = float(np.max(prediction_vector))
        
#         logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
        
#         return {"predicted_class": predicted_class}
        
#     except Exception as e:
#         logger.error(f"Prediction failed: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to process file or make prediction: {str(e)}"
#         )



# #v4 claude code
# import os
# import io
# import tensorflow as tf
# import joblib
# import numpy as np
# import librosa
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import logging

# # Import custom modules
# import config
# import data_processing

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # FastAPI app instance
# app = FastAPI(
#     title="Urban Sound Classification API",
#     description="Audio classification API for Urban Sound dataset",
#     version="1.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # Add CORS middleware for web client access
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global variables
# ml_model = None
# label_encoder = None
# norm_stats = None

# # Response models
# class PredictionResponse(BaseModel):
#     predicted_class: str

# class HealthResponse(BaseModel):
#     status: str
#     message: str

# @app.on_event("startup")
# async def load_artifacts():
#     """Load model artifacts on startup"""
#     global ml_model, label_encoder, norm_stats
    
#     logger.info("🚀 Loading model and artifacts from local directory...")
    
#     try:
#         # Define artifact paths
#         artifacts_dir = "./artifacts"
#         model_path = os.path.join(artifacts_dir, "model.keras")
#         le_path = os.path.join(artifacts_dir, "label_encoder.joblib")
#         stats_path = os.path.join(artifacts_dir, "normalization_stats.joblib")
        
#         # Verify files exist
#         for path, name in [(model_path, "model"), (le_path, "label encoder"), (stats_path, "normalization stats")]:
#             if not os.path.exists(path):
#                 raise FileNotFoundError(f"{name} file not found at {path}")
        
#         # Load artifacts
#         ml_model = tf.keras.models.load_model(model_path)
#         label_encoder = joblib.load(le_path)
#         norm_stats = joblib.load(stats_path)
        
#         logger.info("✅ Model and artifacts loaded successfully!")
#         logger.info(f"Model input shape: {ml_model.input_shape}")
#         logger.info(f"Available classes: {len(label_encoder.classes_)}")
        
#     except Exception as e:
#         logger.error(f"❌ Error loading artifacts: {e}")
#         raise RuntimeError(f"Could not load model or artifacts: {e}")

# @app.get("/", response_model=HealthResponse)
# async def root():
#     """Root endpoint"""
#     return {
#         "status": "success", 
#         "message": "Urban Sound Classification API is running! Visit /docs for API documentation."
#     }

# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """Health check endpoint for monitoring"""
#     if ml_model is None or label_encoder is None or norm_stats is None:
#         raise HTTPException(
#             status_code=503,
#             detail="Service unavailable - model artifacts not loaded"
#         )
#     return {
#         "status": "healthy",
#         "message": "All systems operational - ready for predictions"
#     }

# @app.get("/info")
# async def get_model_info():
#     """Get information about the loaded model"""
#     if ml_model is None or label_encoder is None:
#         raise HTTPException(
#             status_code=503,
#             detail="Model not loaded"
#         )
    
#     return {
#         "model_input_shape": str(ml_model.input_shape),
#         "available_classes": label_encoder.classes_.tolist(),
#         "sample_rate": config.SAMPLE_RATE,
#         "duration": config.DURATION
#     }

# @app.post("/predict", response_model=PredictionResponse)
# async def predict(file: UploadFile = File(...)):
#     """
#     Audio classification endpoint
    
#     Accepts audio files and returns predicted urban sound class
#     """
#     # Check if model is loaded
#     if not ml_model or not label_encoder or not norm_stats:
#         raise HTTPException(
#             status_code=503,
#             detail="Model is not loaded yet. Please wait for initialization."
#         )
    
#     # Validate file type
#     allowed_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')
#     if not file.filename.lower().endswith(allowed_extensions):
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid file format. Supported formats: {', '.join(allowed_extensions)}"
#         )
    
#     # Log prediction request
#     logger.info(f"Processing prediction for file: {file.filename}")
    
#     try:
#         # Read uploaded audio file
#         audio_bytes = await file.read()
#         logger.info(f"File size: {len(audio_bytes)} bytes")
        
#         # Load audio data from memory
#         audio, _ = librosa.load(
#             io.BytesIO(audio_bytes), 
#             sr=config.SAMPLE_RATE, 
#             duration=config.DURATION
#         )
#         logger.info(f"Audio loaded: {len(audio)} samples at {config.SAMPLE_RATE}Hz")
        
#         # Extract MFCC features
#         mfccs = data_processing.extract_mfcc(audio)
#         logger.info(f"MFCC features extracted: shape {mfccs.shape}")
        
#         # Normalize features
#         mfccs_normalized = (mfccs - norm_stats['mean']) / (norm_stats['std'] + 1e-8)
        
#         # Reshape for model input (batch, time_steps, n_mfcc, channels)
#         mfccs_reshaped = mfccs_normalized[np.newaxis, ..., np.newaxis]
#         logger.info(f"Input shape for model: {mfccs_reshaped.shape}")
        
#         # Make prediction
#         prediction_vector = ml_model.predict(mfccs_reshaped, verbose=0)
        
#         # Post-process prediction
#         predicted_index = np.argmax(prediction_vector, axis=1)[0]
#         predicted_class = label_encoder.inverse_transform([predicted_index])[0]
#         confidence = float(np.max(prediction_vector))
        
#         logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
        
#         return {"predicted_class": predicted_class}
        
#     except Exception as e:
#         logger.error(f"Prediction failed: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to process file or make prediction: {str(e)}"
#         )
# #v3
# import os
# import io
# import tensorflow
# import joblib
# import numpy as np
# import librosa # You'll need this for audio processing
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel
# # --- CHANGE 1: Import the Keras load_model function ---
# import tensorflow as tf

# # --- CHANGE 2: MLflow imports are no longer needed ---
# # from mlflow.tracking import MlflowClient
# # from mlflow.exceptions import MlflowException
# # import mlflow

# # --- Custom Module Imports ---
# # This remains the same
# import config
# import data_processing

# # --- Configuration ---
# # These are no longer needed as we load directly from a local folder
# # REGISTERED_MODEL_NAME = "AudioClassifier"
# # MODEL_ALIAS = "Production"

# # --- 1. Define Application and Global Variables ---
# app = FastAPI(title="Audio Classification API")

# ml_model = None
# label_encoder = None
# norm_stats = None

# # --- 2. Define Request/Response Models ---
# # This remains the same
# class PredictionResponse(BaseModel):
#     predicted_class: str

# # --- 3. Implement a Startup Event to Load the Model ---
# # This entire function is replaced to load artifacts locally.
# @app.on_event("startup")
# def load_artifacts():
#     """
#     Loads the production model and preprocessing artifacts from the local '/app/artifacts'
#     directory inside the container.
#     """
#     global tf, ml_model, label_encoder, norm_stats
#     print("Loading model and artifacts from local directory...")

#     try:
#         # Define the paths to the artifacts inside the container
#         # These paths correspond to the `COPY ./artifacts /app/artifacts` command in your Dockerfile
#         artifacts_dir = "./artifacts"
#         # model_path = os.path.join(artifacts_dir, "model","model.keras") # Path to the saved model folder
#         model_path = os.path.join(artifacts_dir,"model.keras")
#         le_path = os.path.join(artifacts_dir, "label_encoder.joblib")
#         stats_path = os.path.join(artifacts_dir, "normalization_stats.joblib")

#         # Load the artifacts
#         # ml_model = load_model(model_path)
#         # Load the artifacts
#         ml_model = tf.keras.models.load_model(model_path)   # keras model
#         label_encoder = joblib.load(le_path)
#         norm_stats = joblib.load(stats_path)

#         print("✅ Model and artifacts loaded successfully.")

#     except Exception as e:
#         print(f"❌ An error occurred during artifact loading: {e}")
#         # This is a critical error, so we raise it to stop the server from starting improperly
#         raise RuntimeError(f"Could not load model or artifacts: {e}")


# # --- 4. Define the Prediction Endpoint ---
# # The logic inside this function remains almost identical.
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
#         # 1. Load audio data from in-memory bytes
#         audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=config.SAMPLE_RATE, duration=config.DURATION)
        
#         # 2. Extract MFCCs
#         mfccs = data_processing.extract_mfcc(audio)
        
#         # 3. Normalize features using the loaded stats
#         mfccs_normalized = (mfccs - norm_stats['mean']) / (norm_stats['std'] + 1e-8)
        
#         # 4. Reshape for the model's input (batch, time_steps, n_mfcc, channels)
#         mfccs_reshaped = mfccs_normalized[np.newaxis, ..., np.newaxis]

#         # --- Make Prediction ---
#         # Use the predict method of the loaded Keras model
#         prediction_vector = ml_model.predict(mfccs_reshaped)
        
#         # --- Post-process the Prediction ---
#         predicted_index = np.argmax(prediction_vector, axis=1)[0]
#         predicted_class = label_encoder.inverse_transform([predicted_index])[0]

#         return {"predicted_class": predicted_class}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to process file or make prediction: {e}")









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

