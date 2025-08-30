
# # Configuration
# DATASET_PATH =r"C:/Users/home/urbnmlops/urbnmlops/urbansound8k"
# METADATA_PATH = r"C:/Users/home/urbnmlops/urbnmlops/urbansound8k/UrbanSound8K.csv"
# SAMPLE_RATE = 22050
# DURATION = 4  # seconds
# N_MFCC = 40
# MAX_TIME_STEPS = 174  # From search result [3]
# #First example change

from pathlib import Path

# --- Core Paths ---
# This automatically finds the root directory of your project
# (assuming config.py is in the root).
# PROJECT_ROOT = Path(__file__).resolve().parent

# # Define data paths relative to the project root
# DATA_DIR = PROJECT_ROOT / "urbansound8k"
# METADATA_PATH = DATA_DIR / "UrbanSound8K.csv"
#The reason i' defining path directly is because the above path code with automatically fetch the path of the data if the data is present in the same directory as the code files, but in my case as there is no storage in my drive i can't copy the dataset again in this path (lack of storage- long story short)
# DATA_DIR =r"C:/Users/home/urbnmlops/urbnmlops/urbansound8k"
# METADATA_PATH = r"C:/Users/home/urbnmlops/urbnmlops/urbansound8k/UrbanSound8K.csv"
# DATA_DIR =r"E:/urban soundscape project/urbansound8k"
# METADATA_PATH = r"E:/urban soundscape project/urbansound8k/UrbanSound8K.csv"
# Old (absolute Windows path):
# DATA_DIR = r'C:/users/home/FtoL/data/urbansound8k/'
# METADATA_PATH = r'C:/users/home/FtoL/data/urbansound8k/UrbanSound8K.csv'

# New (Docker volume path):
# DATA_DIR = '/dataset/'
# METADATA_PATH = '/dataset/UrbanSound8K.csv'

# # New (Continous integration path):
DATA_DIR = 'data/urbansound8k/'
METADATA_PATH = 'data/urbansound8k/UrbanSound8K.csv'


# --- Model & Feature Parameters ---
SAMPLE_RATE = 22050
DURATION = 4  # seconds
N_MFCC = 40
MAX_TIME_STEPS = 175

# --- Training Parameters ---
EPOCHS = 10 # Example of a training parameter
BATCH_SIZE = 32 # Example of a training parameter

MODEL_ARTIFACT_NAME = "model"
LABEL_ENCODER_ARTIFACT_NAME = "label_encoder.joblib"
NORMALIZATION_STATS_ARTIFACT_NAME = "normalization_stats.joblib"