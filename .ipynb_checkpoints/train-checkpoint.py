import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

# Import your custom modules
import config
import data_processing
import model

# --- 1. Load and Prepare Data ---
print("Loading data and extracting features...")
metadata = pd.read_csv(config.METADATA_PATH)
features = []
labels = []

for index, row in metadata.iterrows():
    file_path = os.path.join(config.DATA_DIR, f"fold{row['fold']}", row['slice_file_name'])
    result = data_processing.load_audio_data(file_path)
    if result is not None:
        audio = result
        mfcc = data_processing.extract_mfcc(audio)
        features.append(mfcc)
        labels.append(row['class'])

X = np.array(features)
y = np.array(labels)

# --- 2. Preprocess Data ---
mean = np.mean(X, axis=(0, 1))
std = np.std(X, axis=(0, 1))
X = (X - mean) / (std + 1e-8)
X = X[..., np.newaxis]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
num_classes = len(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42
)

## FIX 1: Define the input_example using your training data
input_example = X_train[:5]

## FIX 2: Define the registered model name in one place
REGISTERED_MODEL_NAME = "AudioClassifier"


# --- 3. Save Preprocessing Artifacts ---
joblib.dump(le, config.LABEL_ENCODER_ARTIFACT_NAME)
joblib.dump({'mean': mean, 'std': std}, config.NORMALIZATION_STATS_ARTIFACT_NAME)

# --- 4. MLflow Experiment Tracking ---
mlflow.set_experiment("Final-sound-cls-model")

with mlflow.start_run() as run:
    print(f"Starting MLflow Run ID: {run.info.run_id}")
    run_id = run.info.run_id

    mlflow.log_params({
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "n_mfcc": config.N_MFCC,
        "test_split_ratio": 0.2
    })

    input_shape = X_train.shape[1:]
    cnn_model = model.build_model(input_shape=input_shape, num_classes=num_classes)
    cnn_model.summary()

    print("\nTraining the model...")
    history = cnn_model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )

    print("\nEvaluating the model...")
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    mlflow.log_metrics({
        "test_accuracy": test_acc,
        "test_loss": test_loss
    })

    for epoch in range(config.EPOCHS):
        mlflow.log_metrics({
            'train_loss': history.history['loss'][epoch],
            'train_accuracy': history.history['accuracy'][epoch],
            'val_loss': history.history['val_loss'][epoch],
            'val_accuracy': history.history['val_accuracy'][epoch]
        }, step=epoch)

    # Log the model as an artifact WITHOUT registering it yet
    mlflow.tensorflow.log_model(cnn_model, "model", input_example=input_example)

    # Log preprocessing artifacts
    mlflow.log_artifact(config.LABEL_ENCODER_ARTIFACT_NAME, "preprocessing")
    mlflow.log_artifact(config.NORMALIZATION_STATS_ARTIFACT_NAME, "preprocessing")

    # --- 5. Conditional Registration Logic ---
    print("\n--- Starting Conditional Model Registration ---")
    client = MlflowClient()
    previous_accuracy = -1.0 # Initialize with a value lower than any possible accuracy

    try:
        # Get the latest version of the model with the 'Production' alias
        latest_prod_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        if latest_prod_versions:
            latest_prod_version = latest_prod_versions[0]
            prod_run = client.get_run(latest_prod_version.run_id)
            
            ## FIX 3: Use the correct metric key "test_accuracy"
            previous_accuracy = prod_run.data.metrics.get("test_accuracy", -1.0)
            print(f"Found existing 'Production' model (Version: {latest_prod_version.version}) with accuracy: {previous_accuracy:.4f}")
        else:
            print("No 'Production' model found. This will be the first one.")

    except Exception as e:
        print(f"No registered model named '{REGISTERED_MODEL_NAME}' found. Registering as first version. Error: {e}")

    ## FIX 4: Perform the comparison and register/promote if better
    if test_acc > previous_accuracy:
        print(f"New model accuracy ({test_acc:.4f}) is better than previous ({previous_accuracy:.4f}).")
        print("Registering new model version and setting alias to 'Production'...")

        # Register the model logged earlier in this run
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=REGISTERED_MODEL_NAME
        )
        
        # Set the alias for the new version
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="Production",
            version=result.version
        )
        print(f"✅ Successfully registered Version {result.version} and set 'Production' alias.")

    else:
        print(f"New model accuracy ({test_acc:.4f}) is not better than the current 'Production' model ({previous_accuracy:.4f}).")
        print("❌ Model will not be registered or promoted.")