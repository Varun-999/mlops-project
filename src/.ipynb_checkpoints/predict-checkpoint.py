
# In predict.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Import the processing functions we just defined
import data_processing 

# --- Paths to Saved Artifacts ---
# These would be loaded from a config file or passed as arguments
MODEL_PATH = 'urban_sound_model.h5'
LABEL_ENCODER_PATH = 'label_encoder.joblib'
NORMALIZATION_STATS_PATH = 'normalization_stats.joblib' # To save mean/std

def predict_sound(audio_file_path):
    """
    Loads a trained model and makes a prediction on a single audio file.

    Args:
        audio_file_path (str): The path to the audio file to classify.

    Returns:
        str: The predicted class name, or an error message.
    """
    try:
        # 1. Load the trained model and artifacts
        model = load_model(MODEL_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        norm_stats = joblib.load(NORMALIZATION_STATS_PATH)
        mean = norm_stats['mean']
        std = norm_stats['std']

        # 2. Load and process the audio file using our functions
        audio = data_processing.load_audio_data(audio_file_path)
        if audio is None:
            return "Error: Could not process audio file."
            
        mfcc = data_processing.extract_mfcc(audio)

        # 3. Normalize and reshape the features exactly as done in training
        mfcc_normalized = (mfcc - mean) / (std + 1e-8)
        mfcc_reshaped = mfcc_normalized.reshape(1, mfcc_normalized.shape[0], mfcc_normalized.shape[1], 1)

        # 4. Make a prediction
        prediction = model.predict(mfcc_reshaped)
        
        # 5. Decode the prediction
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = le.inverse_transform([predicted_index])[0]
        
        return predicted_class_name

    except FileNotFoundError:
        return f"Error: Ensure model ('{MODEL_PATH}'), label encoder ('{LABEL_ENCODER_PATH}'), and stats ('{NORMALIZATION_STATS_PATH}') exist."
    except Exception as e:
        return f"An error occurred: {e}"

# --- Example Usage ---
if __name__ == '__main__':
    # This is a placeholder path. Replace with an actual audio file path.
    # For this to work, you must first run train.py to generate the .h5 and .joblib files.
    EXAMPLE_AUDIO_FILE = "E:\urban soundscape project\urbansound8k\fold1\15564-2-0-0.wav" 

    print(f"Attempting to classify sound from: {EXAMPLE_AUDIO_FILE}")
    
    # Before running, make sure the file exists and the model artifacts are in the same directory.
    import os
    if not os.path.exists(EXAMPLE_AUDIO_FILE):
        print("\n---")
        print(f"WARNING: Example audio file not found at '{EXAMPLE_AUDIO_FILE}'.")
        print("Please replace this placeholder path with a real .wav file to test the prediction script.")
        print("You must also have the trained model artifacts ('urban_sound_model.h5', 'label_encoder.joblib', 'normalization_stats.joblib') available.")
        print("---\n")
    else:
        predicted_label = predict_sound(EXAMPLE_AUDIO_FILE)
        print(f"\nPredicted Sound Class: {predicted_label}")
