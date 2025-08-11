# In data_processing.py
import os
import librosa
import numpy as np
import pandas as pd

# Note: These constants would ideally be loaded from config.py
SAMPLE_RATE = 22050
DURATION = 4  # seconds
N_MFCC = 40
MAX_TIME_STEPS = 174

def load_audio_data(file_path):
    """
    Loads an audio file, pads/trims it to a fixed duration.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        np.ndarray: The loaded and processed audio waveform.
    """
    try:
        audio, sr = librosa.load(file_path, 
                                 sr=SAMPLE_RATE, 
                                 duration=DURATION, 
                                 res_type='kaiser_fast')
        
        # Pad or trim the audio to ensure consistent length
        required_length = SAMPLE_RATE * DURATION
        if len(audio) < required_length:
            audio = np.pad(audio, (0, required_length - len(audio)), mode='constant')
        elif len(audio) > required_length:
            audio = audio[:required_length]
            
        return audio
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

def extract_mfcc(audio):
    """
    Extracts MFCC features from an audio waveform.

    Args:
        audio (np.ndarray): The audio waveform.

    Returns:
        np.ndarray: The extracted MFCC features, transposed to (time_steps, n_mfcc).
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    
    # Pad or trim to a fixed number of time steps
    if mfccs.shape[1] < MAX_TIME_STEPS:
        mfccs = np.pad(mfccs, ((0, 0), (0, MAX_TIME_STEPS - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :MAX_TIME_STEPS]
        
    return mfccs.T  # Transpose to get shape (time_steps, n_mfcc)

def load_and_process_data(metadata_path, dataset_path):
    """
    Loads metadata, processes all audio files, and combines features and labels.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        dataset_path (str): Path to the root directory of the audio dataset.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The feature matrix (X).
            - np.ndarray: The label vector (y).
    """
    metadata = pd.read_csv(metadata_path)
    features = []
    labels = []

    print("Processing all audio files...")
    for index, row in metadata.iterrows():
        file_name = row['slice_file_name']
        fold = row['fold']
        class_label = row['class']
        # file_path = os.path.join(dataset_path, f"fold{fold}", file_name)
        file_path = config.DATA_DIR / f"fold{row['fold']}" / row['slice_file_name']
        
        audio = load_audio_data(file_path)
        if audio is not None:
            mfcc = extract_mfcc(audio)
            features.append(mfcc)
            labels.append(class_label)

    # Convert lists to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    return X, y

