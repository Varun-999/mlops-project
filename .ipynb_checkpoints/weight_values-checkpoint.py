import numpy as np
import mlflow
# --- Parameters ---
MODEL_NAME = "AudioClassifier"
MODEL_STAGE = "Production"

# --- Load the Model from the Registry ---
print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

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
        # Uncomment the line below to see the actual weight values
        # print(f"  Weights values (first 5): \n{weights[0].flatten()[:5]}")
        
        if len(weights) > 1:
            print(f"  Biases shape: {weights[1].shape}")
            # Uncomment the line below to see the actual bias values
            # print(f"  Biases values (first 5): {weights[1][:5]}")