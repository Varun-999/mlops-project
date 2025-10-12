# UrbanSound8K Audio Classification MLOps Pipeline

End-to-end machine learning pipeline for audio classification using the UrbanSound8K dataset. This project features training, evaluation, prediction API (FastAPI), artifact management, and CI/CD workflows with Docker and MLflow integration.

---

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Testing](#tests)
- [Built Using](#built_using)
- [Authors](#authors)
- [Acknowledgements](#acknowledgement)

---

## 🧐 About <a name="about"></a>

This repository provides a robust MLOps workflow for UrbanSound8K audio classification. The pipeline covers:

- Dataset preprocessing
- CNN model training
- Artifact logging (MLflow)
- Inference API (FastAPI)
- Containerization (Docker)
- Data versioning (DVC)
- CI/CD automation (GitHub Actions, Google cloud run)

All steps are containerized for reproducibility and easy deployment.

---
<!-- ![Project Overview](Architectture diagram.png) -->
<img src="https://github.com/Varun-999/mlops-project/blob/main/images/Architecture_diagram.png" alt="Project Overview" width="600"/>
<!-- ![Architecture Diagram](https://github.com/Varun-999/mlops-project/blob/main/images/Architecture_diagram.png) -->
<!-- <img src="Architecture diagram.png" alt="Project Overview" width="600"/> -->
---

## 🏁 Getting Started <a name="getting_started"></a>

### Prerequisites

- Python 3.11
- pip
- Docker & Docker Compose
- UrbanSound8K dataset (CSV metadata and audio folders)
- (Optional) Google Drive credentials for DVC pull

---

## 🔧 Installation <a name="installation"></a>

1. Clone the repository and navigate to the project directory.
2. Place the UrbanSound8K dataset and metadata CSV in the appropriate `data/` directory.
3. Install Python dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Build and start all services (API, MLflow server, training jobs):

    ```sh
    docker compose up --build
    ```

5. Pull DVC-managed data (if needed):

    ```sh
    dvc pull -v data/urbansound8k/fold1.dvc
    ```

6. Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs) after containers start.

---

## 🎈 Usage <a name="usage"></a>

- **Train a new CNN model:**

    ```sh
    python src/train.py
    ```

- **Serve the prediction API:**

    ```sh
    uvicorn src.main:app --reload
    ```

- **Predict sound class from a local file:**

    ```sh
    python src/predict.py --file path/to/audio.wav
    ```

All model artifacts required for deployment are logged by MLflow and loaded at service start.

---

## 🚀 Deployment <a name="deployment"></a>

For live deployment, use the provided `docker-compose.yml`:

```sh
docker compose up --build
```

This starts:

- MLflow tracking server
- FastAPI audio classification service
- Dedicated training job container

Host volumes are used for model artifacts, DVC-managed data, and MLflow runs, ensuring reproducibility and persistence.

---

## 🔧 Running the tests <a name="tests"></a>

To run automated pipeline tests:

```sh
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
docker compose run --rm train-job
```

For code style checks (manual addition recommended):

```sh
flake8 .
```

---

## ⛏️ Built Using <a name="built_using"></a>

- [FastAPI](https://fastapi.tiangolo.com/) - API Server
- [TensorFlow](https://www.tensorflow.org/) - Deep Learning Framework
- [scikit-learn](https://scikit-learn.org/) - Label Encoding, Preprocessing
- [Librosa](https://librosa.org/) - Audio Feature Extraction
- [MLflow](https://mlflow.org/) - Experiment Tracking & Model Registry
- [Docker](https://www.docker.com/) - Containerization & Deployment
- [DVC](https://dvc.org/) - Data Version Control
- [GitHub Actions](https://github.com/features/actions) - CI/CD Pipeline

---
## Deployed API Link:
https://urbansound-api-205836669252.asia-south1.run.app
Append /docs to the above link to access the api

---
## ✍️ Authors <a name="authors"></a>

- P.Saivarun - Idea & Initial work

---

## 🎉 Acknowledgements <a name="acknowledgement"></a>

- References: FastAPI, Docker, MLflow, DVC documentation

---
