import numpy as np
import mlflow
# # --- Parameters ---
# MODEL_NAME = "AudioClassifier"
# MODEL_ALIAS = "Production"

# # --- Load the Model from the Registry ---
# print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")
# model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"


from mlflow.tracking import MlflowClient

# --- Configuration ---
MODEL_NAME = "AudioClassifier"  # Replace with your model's name
MODEL_ALIAS = "Production"      # The alias you want to load

# --- Load the Model using its Alias ---
print(f"Loading model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'...")
client = MlflowClient()

# 1. Get the specific model version for the given alias
model_version_details = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
version_number = model_version_details.version

# 2. Construct the URI with the explicit version number
model_uri = f"models:/{MODEL_NAME}/{version_number}"





# Load the underlying Keras model
# Note: We use mlflow.tensorflow.load_model here, not pyfunc
keras_model = mlflow.tensorflow.load_model(model_uri)

print("\n--- Model Weights Inspection ---")
# Loop through all the layers in the model
for layer in keras_model.layers:
    # get_weights() returns a list of numpy arrays: [weights, biases]
    weights = layer.get_weights() 
    
    if weights:  # Check if the layer has weights (e.g., Dense, Conv2D)
        print(f"\nLayer: {layer.name}")
        print(f"  Weights shape: {weights[0].shape}")
        print(f"Weights : {weights}")
        # Uncomment the line below to see the actual weight values
        # print(f"  Weights values (first 5): \n{weights[0].flatten()[:5]}")
        
        if len(weights) > 1:
            print(f"  Biases shape: {weights[1].shape}")
            # Uncomment the line below to see the actual bias values
            # print(f"  Biases values (first 5): {weights[1][:5]}")