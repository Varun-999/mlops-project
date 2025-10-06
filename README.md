

<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>
<h3 align="center">UrbanSound8K Audio Classification MLOps Pipeline</h3>

<div align="center">

[
[
[
[

</div>

***

<p align="center">
End-to-end machine learning pipeline for audio classification using the UrbanSound8K dataset, featuring training, evaluation, prediction API (FastAPI), artifact management, and CI/CD workflows with Docker and MLflow integration.
    <br>
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)
- [Built Using](#built_using)
- [TODO](../TODO.md)
- [Contributing](../CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)


## 🧐 About <a name = "about"></a>

This project provides a robust MLOps workflow for UrbanSound8K audio classification tasks. The pipeline covers dataset preprocessing, CNN model training, artifact logging (MLflow), and an inference API using FastAPI. All steps, from feature extraction to model deployment, are containerized with Docker, enabling reproducibility and ease of deployment. CI/CD pipelines run validation, training, and deployment steps using GitHub Actions and DVC for data versioning.[^3][^6][^8][^1]

## 🏁 Getting Started <a name = "getting_started"></a>

Follow these steps to set up the development environment and run the project locally. Refer to the [Deployment](#deployment) section for production deployment notes.

### Prerequisites

- Python 3.11
- pip
- Docker and Docker Compose
- UrbanSound8K dataset (CSV metadata and audio folders)
- (Optional) Google Drive credentials for DVC pull

Install all Python dependencies:

```
pip install -r requirements.txt
```

Other dependencies are specified in the `requirements.txt` and handled by the Docker setup.[^6][^7][^8]

### Installing

1. Clone the repository and navigate to the project directory.
2. Place UrbanSound8K dataset and metadata CSV in the appropriate `dataset/` directory.
3. Copy provided `docker-compose.yml` and `Dockerfile` templates.
4. Build and start all services (API, MLflow server, training jobs):
```
docker compose up --build
```

5. Pull DVC-managed data:
```
dvc pull -v data/urbansound8k/fold1.dvc
```

6. Access the API documentation at http://localhost:8000/docs after containers start.[^7][^8][^6]

## 🔧 Running the tests <a name = "tests"></a>

To run automated pipeline tests through CI/CD:

### End-to-End Tests

These tests validate:

- Environment setup (Python, TensorFlow, FastAPI, joblib, etc.)
- Successful ML training with the provided dataset sample

Example (run by GitHub Actions):

```
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
docker compose run --rm train-job
```


### Coding Style Tests

Lint and code style checks can be added with tools like `flake8` or `black` (not provided out-of-the-box in this repo).

Example:

```
flake8 .
```

(Manual addition recommended for further development).[^8][^3][^6]

## 🎈 Usage <a name="usage"></a>

- Train a new CNN model with the provided UrbanSound8K dataset:

```
python train.py
```

- Serve a prediction API (FastAPI) for audio file inference:

```
uvicorn main:app --reload
```

- Predict sound class from a local file:

```
python predict.py --file path/to/audio.wav
```


All model artifacts required for deployment are logged by MLflow and loaded at service start.[^2][^1]

## 🚀 Deployment <a name = "deployment"></a>

For live system deployment, use the provided `docker-compose.yml`:

```
docker compose up --build
```

This starts:

- An MLflow tracking server
- The FastAPI audio classification service
- A dedicated training job container

Host volumes are used for model artifacts, DVC-managed data, and MLflow runs, ensuring reproducibility and persistence. For production, configure environment variables and externalize database/storage as needed.[^6][^8]

## ⛏️ Built Using <a name = "built_using"></a>

- [FastAPI](https://fastapi.tiangolo.com/) - API Server
- [TensorFlow](https://www.tensorflow.org/) - Deep Learning Framework
- [scikit-learn](https://scikit-learn.org/) - Label Encoding, Preprocessing
- [Librosa](https://librosa.org/) - Audio Feature Extraction
- [MLflow](https://mlflow.org/) - Experiment Tracking \& Model Registry
- [Docker](https://www.docker.com/) - Containerization \& Deployment
- [DVC](https://dvc.org/) - Data Version Control
- [GitHub Actions](https://github.com/features/actions) - CI/CD Pipeline


## ✍️ Authors <a name = "authors"></a>

- [Your Name/Handle] - Idea \& Initial work

See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used
- Inspiration from MLOps best practices and the UrbanSound8K dataset project
- References: FastAPI, Docker, MLflow, DVC documentation

***

Replace `[Your Name/Handle]` with the actual author for proper credit. This README summarizes and organizes instructions and context from your codebase and setup scripts for clarity and onboarding.[^4][^5][^9][^1][^2][^3][^7][^8][^6]

<div align="center">⁂</div>

[^1]: main.py

[^2]: predict.py

[^3]: train.py

[^4]: config.py

[^5]: data_processing.py

[^6]: mlops-ci-cd.yml

[^7]: requirements.txt

[^8]: docker-compose.yml

[^9]: model.py

