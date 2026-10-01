# Control Valve Stiction Detection using OT Images and Deep CORAL

This repository contains the deployment implementation of a control-valve stiction detection framework based on **Optimal Transport (OT) images, Deep CORAL, and Logistic Regression**.

The main goal of this project is to show how a research model can be converted into a practical application using **FastAPI, Streamlit, Docker, Docker Compose, and Kubernetes**.

## Live Demo

**Streamlit application:**  
https://stiction-deep-coral.streamlit.app

**FastAPI documentation:**  
https://stiction-deep-coral-api.onrender.com/docs

**GitHub repository:**  
https://github.com/seshu-damarla/stiction-deep-coral-mlops

---

## Research Paper

This repository is based on the following research paper:

**Optimal Transport Image Representation and Deep Covariance Alignment (CORAL) for Control Valve Stiction Detection**

**Author:** Seshu K. Damarla

**arXiv:** https://arxiv.org/abs/2607.22486  
**DOI:** https://doi.org/10.48550/arXiv.2607.22486

The paper addresses an important problem in data-driven stiction detection. A model trained only with simulated control-loop data may not perform well on real industrial data because the simulated and industrial data have different distributions. This difference is treated as a domain-shift problem.

The proposed method combines **Optimal Transport imaging** and **Deep CORAL domain adaptation**. Controller output (OP) and process variable (PV) signals are converted into two-dimensional OT images. A CNN encoder then learns useful features from these images.

During training, the encoder uses two types of information:

- labeled OT images generated from simulation data,
- unlabeled OT images from industrial control loops.

The training objective combines a classification loss on the labeled simulation data with a Deep CORAL loss. The CORAL loss aligns the covariance of the source-domain and target-domain feature distributions. This helps the encoder learn features that are less dependent on whether the data come from simulation or an industrial process.

After domain adaptation, the learned features are used for final stiction classification.

The paper evaluated the method on an independent set of **20 industrial benchmark control loops**. The proposed method correctly diagnosed **18 out of 20 loops**. It detected all **13 stiction cases**, giving:

| Metric | Result |
|---|---:|
| Accuracy | 90.00% |
| Precision | 86.67% |
| Recall | 100.00% |
| F1-score | 92.86% |

The results show that domain adaptation can reduce the gap between simulation data and industrial data and improve the practical use of data-driven stiction detection.

### Citation

If you use this repository or the proposed method, please cite the paper:

```bibtex
@article{Damarla2026OTDeepCORAL,
  title   = {Optimal Transport Image Representation and Deep Covariance Alignment (CORAL) for Control Valve Stiction Detection},
  author  = {Damarla, Seshu K.},
  journal = {arXiv preprint arXiv:2607.22486},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.22486}
}
```

---

## Project Overview

Control-valve stiction is a common problem in industrial process control loops. It can cause oscillations, poor control performance, and unnecessary process variability.

In this work, process signals are converted into OT images. A CNN encoder trained using Deep CORAL is used to extract domain-adapted features. These features are then classified using Logistic Regression.

The final inference path is:

```text
OT Image
   ↓
Deep CORAL CNN Encoder
   ↓
64-dimensional feature vector
   ↓
StandardScaler
   ↓
Logistic Regression
   ↓
Stiction / Non-stiction
```

The deployed model uses the trained Deep CORAL encoder and the final Logistic Regression classifier from the research study.

---

## Main Features

- Deep CORAL based feature extraction
- Logistic Regression classifier
- FastAPI REST API
- Streamlit web interface
- Docker containerization
- Docker Compose for running frontend and backend together
- Kubernetes deployment
- Health and model-information API endpoints
- Automated tests for preprocessing, model loading, inference, and API endpoints
- Public Streamlit demo
- Public FastAPI backend

---

## Model Performance

The final model was evaluated on 20 industrial test loops.

| Metric | Value |
|---|---:|
| Accuracy | 0.9000 |
| Precision | 0.8667 |
| Recall | 1.0000 |
| F1-score | 0.9286 |

Confusion matrix:

```text
[[5, 2],
 [0, 13]]
```

The model correctly detected all 13 stiction loops in the test set.

---

## Repository Structure

```text
stiction-deep-coral-mlops/
│
├── artifacts/
│   ├── encoder.pt
│   ├── logistic_regression.joblib
│   ├── threshold.json
│   └── model_metadata.json
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── inference.py
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── frontend/
│   └── app.py
│
├── tests/
│
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.frontend
│
├── kubernetes/
│   ├── api-deployment.yml
│   ├── api-service.yml
│   ├── frontend-deployment.yml
│   └── frontend-service.yml
│
├── demo_data/
│
├── docker-compose.yml
├── requirements.txt
├── requirements-api.txt
├── requirements-frontend.txt
└── README.md
```

---

## FastAPI Backend

The FastAPI application provides the following endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API status |
| `/health` | GET | Check API and model status |
| `/model-info` | GET | Show model information |
| `/predict` | POST | Upload an OT image and obtain a prediction |

Example health response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

The public API documentation is available at:

https://stiction-deep-coral-api.onrender.com/docs

---

## Streamlit Application

The Streamlit interface allows the user to:

1. upload an OT image,
2. send the image to the FastAPI service,
3. obtain the stiction probability,
4. view the final stiction or non-stiction diagnosis.

Public application:

https://stiction-deep-coral.streamlit.app

---

## Run Locally

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Start Streamlit in another terminal:

```bash
streamlit run frontend/app.py
```

The Streamlit application will be available at:

```text
http://127.0.0.1:8501
```

---

## Run with Docker Compose

Build and start both services:

```bash
docker compose up --build
```

The services are:

```text
Streamlit frontend  → http://localhost:8501
FastAPI backend     → http://localhost:8001
```

Stop the services using:

```bash
docker compose down
```

---

## Kubernetes Deployment

The application was also deployed locally using Kubernetes.

The Kubernetes architecture is:

```text
Browser
   ↓
Streamlit Service
   ↓
Streamlit Pod
   ↓
FastAPI Service
   ↓
FastAPI Pod
   ↓
Deep CORAL + Logistic Regression
```

Apply the API deployment and service:

```bash
kubectl apply -f kubernetes/api-deployment.yml
kubectl apply -f kubernetes/api-service.yml
```

Apply the Streamlit deployment and service:

```bash
kubectl apply -f kubernetes/frontend-deployment.yml
kubectl apply -f kubernetes/frontend-service.yml
```

Check the deployment:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

To access Streamlit locally:

```bash
kubectl port-forward service/stiction-frontend-service 8501:8501
```

Then open:

```text
http://127.0.0.1:8501
```

---

## Public Deployment

For the public demonstration:

```text
User
  ↓
Streamlit Community Cloud
  ↓
FastAPI on Render
  ↓
Deep CORAL Encoder
  ↓
Logistic Regression
  ↓
Stiction / Non-stiction
```

The Streamlit application is hosted on **Streamlit Community Cloud**, and the FastAPI backend is hosted on **Render**.

---

## Technologies Used

- Python
- PyTorch
- scikit-learn
- FastAPI
- Streamlit
- Docker
- Docker Compose
- Kubernetes
- pytest
- Render
- Streamlit Community Cloud

---

## Research Context

This deployment is based on the research paper:

**Optimal Transport Image Representation and Deep Covariance Alignment (CORAL) for Control Valve Stiction Detection**  
https://arxiv.org/abs/2607.22486

The paper focuses on the development and evaluation of the stiction-detection method. This repository mainly focuses on the **deployment and MLOps side** of the same research model.

The purpose of the repository is to show how the trained research model can be moved from an experimental notebook to a practical application. The model is packaged as reusable inference artifacts, exposed through FastAPI, connected to a Streamlit interface, containerized with Docker, deployed using Docker Compose and Kubernetes, and finally made available through a public web application.

---

## Author

**Seshu Kumar Damarla**

Research interests include process control, process systems engineering, industrial AI, soft sensors, predictive maintenance, and data-driven process monitoring.
