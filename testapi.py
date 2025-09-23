import requests

# Test the health endpoint
response = requests.get("https://urbansound-classifier.onrender.com/health")
print("Health:", response.json())

# Test the info endpoint
response = requests.get("https://urbansound-classifier.onrender.com/info")
print("Info:", response.json())

# Test prediction (replace with your audio file path)
with open("C:/Users/home/FtoL/data/urbansound8k/fold1/7061-6-0-0.wav", "rb") as f:
    files = {"file": f}
    response = requests.post("https://urbansound-classifier.onrender.com/predict", files=files)
    if response.status_code == 200:
        print("Prediction:", response.json())
    else:
        print("Error:", response.status_code, response.text)