import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import mlflow
import mlflow.tensorflow

# Import your custom modules
# These would contain your file paths and model hyperparameters
# e.g., config.EPOCHS, config.BATCH_SIZE, etc.
import config 
import data_processing
import model

# --- 1. Load and Prepare Data ---
# This section is run once, before the experiment starts.

print("Loading data and extracting features...")
metadata = pd.read_csv(config.METADATA_PATH)
features = []
labels = []

for index, row in metadata.iterrows():
    # Construct the full file path
    file_path = os.path.join(config.DATA_DIR, f"fold{row['fold']}", row['slice_file_name'])
    
    # Load audio and handle potential loading errors
    result = data_processing.load_audio_data(file_path)
    
    if result is not None:
        audio = result
        # Extract features for the loaded audio
        mfcc = data_processing.extract_mfcc(audio)
        
        features.append(mfcc)
        labels.append(row['class'])

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# --- 2. Preprocess Data ---

# Normalize features
# Calculate mean and std for normalization
mean = np.mean(X, axis=(0, 1))
std = np.std(X, axis=(0, 1))
X = (X - mean) / (std + 1e-8) # Add epsilon to avoid division by zero

# Reshape for CNN input (adding a channel dimension)
X = X[..., np.newaxis]

# Encode string labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
num_classes = len(le.classes_)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42
)

# --- 3. Save Preprocessing Artifacts ---
# Save the label encoder and normalization stats. These are essential for making
# predictions on new data later. We save them before the run so we can log them
# as artifacts within the run.
joblib.dump(le, config.LABEL_ENCODER_ARTIFACT_NAME)
joblib.dump({'mean': mean, 'std': std}, config.NORMALIZATION_STATS_ARTIFACT_NAME)

# --- 4. MLflow Experiment Tracking ---

# Set the experiment name. If it doesn't exist, MLflow creates it.
mlflow.set_experiment("Final-sound-cls-model")

# Start a new MLflow run. The 'with' statement ensures the run is properly closed.
with mlflow.start_run() as run:
    print(f"Starting MLflow Run ID: {run.info.run_id}")

    # Log hyperparameters from your config file
    mlflow.log_params({
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        # "learning_rate": config.LEARNING_RATE, # Assuming you have this in config
        "n_mfcc": config.N_MFCC,
        "test_split_ratio": 0.2
    })

    # Build the model using the function from model.py
    input_shape = X_train.shape[1:]
    cnn_model = model.build_model(input_shape=input_shape, num_classes=num_classes)
    cnn_model.summary()

    # Train the model
    print("\nTraining the model...")
    history = cnn_model.fit(
        X_train, 
        y_train,
        validation_data=(X_test, y_test),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=1 # Set to 1 to see progress
    )

    # Evaluate the model on the test set
    print("\nEvaluating the model...")
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    # Log evaluation metrics
    mlflow.log_metrics({
        "test_accuracy": test_acc,
        "test_loss": test_loss
    })
    
    # Log the training history metrics (loss, accuracy, val_loss, val_accuracy)
    for epoch in range(config.EPOCHS):
        mlflow.log_metrics({
            'train_loss': history.history['loss'][epoch],
            'train_accuracy': history.history['accuracy'][epoch],
            # 'val_loss': history.history['validation_loss'][epoch],
            # 'val_accuracy': history.history['validation_accuracy'][epoch]
        }, step=epoch)


    # Log the model itself
    # This saves the model in a format that MLflow understands, making it easy to deploy.
    mlflow.tensorflow.log_model(cnn_model, "model")

    # Log the preprocessing artifacts we saved earlier
    mlflow.log_artifact(config.LABEL_ENCODER_ARTIFACT_NAME, "preprocessing")
    mlflow.log_artifact(config.NORMALIZATION_STATS_ARTIFACT_NAME, "preprocessing")

    print("\n✅ Run completed and all artifacts logged.")

