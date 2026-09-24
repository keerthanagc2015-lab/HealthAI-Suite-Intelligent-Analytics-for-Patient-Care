# HealthAI — Intelligent Healthcare AI Platform

HealthAI is an end-to-end healthcare AI portfolio platform integrating classical machine learning, deep learning, medical NLP, retrieval-augmented generation (RAG), agentic orchestration, multilingual patient interaction, FastAPI, Streamlit, and Docker.

> **Important:** HealthAI is an educational and portfolio demonstration. It is not a medical device and is not intended for diagnosis, treatment decisions, or clinical decision-making.

---

## Overview

HealthAI brings multiple healthcare AI workflows together in one modular application.

### Core capabilities

- Healthcare risk and classification workflows
- Hospital length-of-stay prediction
- Primary diagnosis prediction
- Patient clustering
- Association-rule mining
- CNN-based chest X-ray analysis
- RNN/LSTM patient deterioration workflow
- Patient feedback sentiment analysis
- Medical named-entity recognition (NER)
- Evidence-grounded medical RAG
- Agentic AI routing and tool execution
- Multilingual patient interaction
- English-to-Hindi neural machine translation
- FastAPI REST services
- Streamlit user interface
- Docker-based deployment

---

## Architecture

```text
User
 │
 ▼
Streamlit UI
 │
 ▼
FastAPI REST API
 │
 ▼
Agent Router / Executor
 │
 ├── Classification tools
 ├── Regression tools
 ├── Clustering / Association tools
 ├── CNN / RNN-LSTM tools
 ├── Sentiment / NER tools
 ├── Medical RAG
 └── Translation
 │
 ▼
Models / Retrieved Evidence
 │
 ▼
Normalized API Response
 │
 ▼
Streamlit UI
```

---

## Project Structure

```text
HealthAI-Suite-Intelligent-Analytics-for-Patient-Care/
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements-runtime.txt
│
├── docs/
│   └── classification/
│       └── classification_results.md
│
├── notebooks/
│   └── 01_Exploratory_Data_Analysis.ipynb
│
├── src/
│   ├── 01_data_loader.py
│   ├── 02_data_understanding.py
│   ├── 03_data_quality.py
│   ├── 04_data_cleaning.py
│   ├── 05_feature_selection.py
│   ├── 06_feature_engineering.py
│   ├── 07_baseline_model_training.py
│   ├── 08_linear_svc_experiment.py
│   ├── 09_advanced_feature_engineering.py
│   │
│   ├── 10_advanced_model_training/
│   ├── 11_hyperparameter_tuning/
│   ├── 12_feature_importance/
│   ├── 13_length_of_stay_regression/
│   ├── 14_patient_segmentation/
│   ├── 15_associative_learning/
│   ├── 16_deep_learning/
│   ├── 17_agentic_ai/
│   ├── 18_api/
│   └── 19_streamlit/
│
├── test/
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
└── README.md
```

---

# AI / ML Modules

| Module | Approach | Purpose |
|---|---|---|
| Healthcare Classification | Classical Machine Learning | Structured healthcare risk and category prediction |
| Hospital Length of Stay | Regression / XGBoost | Predict hospital stay duration |
| Primary Diagnosis | Classification | Predict configured diagnosis categories |
| Patient Clustering | Unsupervised Learning | Segment patient records into data-driven groups |
| Association Mining | Apriori / Association Rules | Identify recurring feature relationships |
| Chest X-ray | CNN | Normal / pneumonia image classification |
| Patient Deterioration | RNN / LSTM | Model sequential deterioration patterns |
| Sentiment Analysis | Transformer / DistilBERT workflow | Classify patient feedback sentiment |
| Medical NER | BioBERT / ClinicalBERT workflow | Extract medical entities from text |
| Medical RAG | Retrieval + Embeddings | Ground medical-information responses in curated sources |
| Agentic AI | Router + Executor + Adapters | Route requests to specialized healthcare tools |
| Medical Translation | Helsinki-NLP NMT | English-to-Hindi translation |

---

# Technology Stack

### Programming

- Python
- SQL
- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost
- Feature engineering
- Feature selection
- Hyperparameter tuning
- Classification
- Regression
- Clustering
- Association-rule mining

### Deep Learning

- TensorFlow
- Keras
- PyTorch
- CNN
- RNN
- LSTM

### NLP

- Transformers
- BioBERT
- ClinicalBERT
- DistilBERT
- Sentence Transformers
- Medical NER
- Sentiment analysis
- Neural machine translation

### AI / Architecture

- Agentic AI
- Tool routing
- Model adapters
- Retrieval-Augmented Generation
- Semantic retrieval
- Question-aware reranking
- Response normalization

### Application

- FastAPI
- Uvicorn
- Streamlit
- Docker
- Docker Compose

---

# Data Sources & Dataset References

The following external datasets were used for different HealthAI workflows.

## Indian Healthcare Patient Records

**Use:** Primary tabular healthcare modelling, classification and structured-data workflows.

**Source:** Kaggle — Arun's Workspace

https://www.kaggle.com/datasets/arunsworkspace/indian-healthcare-patient-records

The dataset contains 100,000 healthcare records used in the project's structured-data workflows.

---

## LengthOfStay.csv

**Use:** Hospital length-of-stay regression and ANN/MLP workflows.

**Source:** Microsoft R Server — Hospital Length of Stay

https://microsoft.github.io/r-server-hospital-length-of-stay/

**License:** MIT

The Microsoft project describes the data as synthetic records modelled after real-world hospital inpatient records.

---

## Patient Feedback and Sentiment Analysis Dataset

**Use:** Patient feedback, sentiment, satisfaction and healthcare service-quality analysis.

**Source:** Kaggle — Patient Feedback and Sentiment Analysis Dataset

https://www.kaggle.com/datasets/sanak2000/cleveland-clinic-patients-feedback

**License:** MIT

**Local file:**

```text
data/raw/medical_sentiment/patient_feedback_dataset.xlsx
```

The dataset used in the project contains 1,000 rows and 5 columns.

---

## Healthcare_DataSet.csv

**Use:** Medical-text and sentiment exploration.

**Local file:**

```text
data/raw/medical_sentiment/Healthcare_DataSet.csv
```

**Records:** 87,057

---

## Hospital Deterioration — Simulated Early Warning

**Use:** RNN/LSTM patient deterioration workflow.

**Source:** Hugging Face — `tarekmasryo/hospital-deterioration-dataset`

https://huggingface.co/datasets/tarekmasryo/hospital-deterioration-dataset

**License:** CC BY 4.0

The dataset is described as simulated healthcare data.

---

## Synthetic Chest X-Ray Pneumonia Dataset

**Use:** CNN normal/pneumonia image classification.

**Source:** Hugging Face — `chimbiwide/synthetic-chest-xray-pneumonia`

https://huggingface.co/datasets/chimbiwide/synthetic-chest-xray-pneumonia

**License:** CC BY 4.0

The project presents this dataset as synthetic and uses it for portfolio model development.

---

## Multilingual Synthetic Medical Notes for NER

**Use:** Medical named-entity recognition.

**Source:** Hugging Face — `E3-JSI/synthetic-multi-med-notes-ner-v1`

https://huggingface.co/datasets/E3-JSI/synthetic-multi-med-notes-ner-v1

**License:** MIT

The dataset is synthetic clinical-style text used for medical NER experimentation.

---

# Dataset Storage

Large datasets and generated artifacts are intentionally excluded from the Git repository.

The project excludes:

```text
models/
mlruns/
data/raw/
data/processed/
```

This keeps large datasets and generated machine-learning artifacts outside Git history.

The datasets should be obtained from their respective source websites when reproducing the project.

---

# Pretrained Models & References

## BioBERT v1.1

**Use:** Biomedical named-entity recognition.

**Source:**

https://huggingface.co/dmis-lab/biobert-v1.1

---

## Bio_ClinicalBERT

**Use:** Clinical NLP and medical NER.

**Source:**

https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

**License:** MIT

---

## DistilBERT Base Uncased

**Use:** Sentiment transformer workflow.

**Source:**

https://huggingface.co/distilbert/distilbert-base-uncased

**License:** Apache-2.0

---

## all-MiniLM-L6-v2

**Use:** Sentence embeddings for the medical RAG workflow.

**Source:**

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

**License:** Apache-2.0

---

## Helsinki-NLP/opus-mt-en-hi

**Use:** English-to-Hindi neural machine translation.

**Source:**

https://huggingface.co/Helsinki-NLP/opus-mt-en-hi

**License:** Apache-2.0

This is a pretrained neural machine-translation model and is not presented as a general-purpose LLM.

---

# Medical RAG

The HealthAI medical-information assistant uses a retrieval-augmented generation architecture based on curated medical sources.

## RAG workflow

```text
User Question
      │
      ▼
Question Embedding
      │
      ▼
Similarity Retrieval
      │
      ▼
Question-Aware Reranking
      │
      ▼
Safety Threshold
      │
      ├── Insufficient Evidence
      │        ↓
      │    Abstention
      │
      └── Sufficient Evidence
               ↓
        Grounded Response
               ↓
          Source Display
```

The current RAG knowledge base contains:

- 55 vectors
- 384-dimensional embeddings
- 8 curated medical sources

The project uses retrieval evidence and a safety threshold before constructing a medical-information response.

---

# RAG Source Websites

The project uses curated public medical-information sources from the following websites.

| Organization | Topic | Source |
|---|---|---|
| World Health Organization | Diabetes | https://www.who.int/health-topics/diabetes |
| WHO India | Diabetes | https://www.who.int/india/health-topics/diabetes |
| World Health Organization | Hypertension | https://www.who.int/health-topics/hypertension/ |
| WHO India | Hypertension | https://www.who.int/india/health-topics/hypertension |
| World Health Organization | Noncommunicable Diseases | https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases |
| MedlinePlus / U.S. National Library of Medicine | Health Topics | https://medlineplus.gov/healthtopics.html |
| MedlinePlus / U.S. National Library of Medicine | Type 2 Diabetes Self-Care | https://medlineplus.gov/ency/patientinstructions/000328.htm |
| MedlinePlus / U.S. National Library of Medicine | Living With Chronic Illness | https://medlineplus.gov/ency/patientinstructions/000602.htm |

The RAG system retains the source URLs used by the project source registry.

---

# Agentic AI

The agentic layer uses a modular router and executor architecture.

```text
User Query
    │
    ▼
Agent Router
    │
    ▼
Selected Healthcare Tool
    │
    ▼
Model / RAG Adapter
    │
    ▼
Structured Result
    │
    ▼
FastAPI
    │
    ▼
Streamlit
```

The agent layer can route requests to specialized healthcare tools instead of sending every request through a single generic model.

---

# Model Adapter Architecture

HealthAI separates model-specific implementation from the application interface.

```text
Streamlit
    │
    ▼
FastAPI
    │
    ▼
Agent Router
    │
    ▼
Tool / Adapter
    │
    ├── Preprocessing
    ├── Model Inference
    ├── Output Normalization
    └── Metadata
```

This architecture allows the individual model components to evolve independently of the presentation layer.

---

# FastAPI

FastAPI provides the REST service layer between the Streamlit interface and the healthcare AI components.

Representative API endpoints include:

```text
GET  /
GET  /health
POST /predict/diabetes
POST /predict/xray
POST /predict/los
POST /predict/diagnosis
```

The API layer provides structured response contracts so that the Streamlit application does not depend directly on model-specific output formats.

---

# Streamlit

Streamlit provides the user-facing application.

The application includes patient-facing and analytics-oriented workflows such as:

- Dashboard
- Health Risk
- HealthAI Assistant
- Appointments
- Hospital Services
- Hospital length-of-stay prediction
- Chest X-ray analysis
- Primary diagnosis prediction
- Patient sentiment analysis
- Medical NER
- Patient clustering
- Association analysis
- Medical RAG
- Multilingual patient interaction

The Patient Portal supports English, Hindi and Tamil interface experiences.

---

# Medical Translation

The medical translation workflow uses:

```text
Helsinki-NLP/opus-mt-en-hi
```

for English-to-Hindi neural machine translation.

The application also contains patient-facing interface translations for English, Hindi and Tamil.

Machine translation is intended for information support and should not replace professional medical interpretation.

---

# Chest X-Ray Module

The chest X-ray workflow uses a CNN-based image classification model.

The application preprocesses uploaded X-ray images before inference and returns a normalized prediction response.

The project presents the X-ray feature as a portfolio demonstration based on a synthetic dataset.

> **Safety:** The X-ray feature is not intended for clinical diagnosis or medical decision-making.

---

# Patient Deterioration

The deterioration workflow uses RNN/LSTM-based sequential modelling.

The workflow is designed to analyse longitudinal or time-series patient signals and model deterioration-related patterns.

The associated dataset is the simulated Hospital Deterioration dataset documented above.

---

# Sentiment Analysis

The patient-feedback workflow analyses healthcare feedback using transformer-based NLP.

The workflow includes:

- Patient feedback inspection
- Sentiment classification
- Satisfaction analysis
- Feedback preprocessing
- Transformer-based modelling
- Error analysis

The project also documents the number of unique feedback texts and the dataset characteristics observed during exploratory analysis.

---

# Medical NER

The Medical NER workflow uses biomedical and clinical transformer resources.

The system includes:

- BioBERT
- Bio_ClinicalBERT
- Synthetic medical notes
- Medical entity extraction
- Entity confidence handling
- Adapter-based inference

---

# Evaluation

The project uses model-specific evaluation approaches.

### Classification

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC where applicable

### Regression

- MAE
- RMSE
- R²

### Clustering

- Cluster profiles
- Cluster interpretability
- Unsupervised evaluation measures where applicable

### Association Rules

- Support
- Confidence
- Lift

### Deep Learning

Model-specific evaluation and prediction workflows are included for CNN, RNN and LSTM components.

### RAG

RAG evaluation focuses on:

- Retrieval quality
- Evidence grounding
- Source usage
- Safety threshold behaviour
- Abstention when sufficient evidence is not available

---

# Docker Deployment

HealthAI includes a Docker deployment configuration under:

```text
deployment/
```

## Docker components

```text
deployment/
├── Dockerfile
├── docker-compose.yml
└── requirements-runtime.txt
```

The project uses a shared application image for the FastAPI and Streamlit services.

Large local `data/` and `models/` directories are mounted at runtime rather than copied into the Docker image.

---

# Run with Docker

From the project root:

```powershell
docker compose -f .\deployment\docker-compose.yml up --build
```

## Application URLs

### Streamlit

```text
http://localhost:8501
```

### FastAPI

```text
http://localhost:8000
```

---

# Run Locally Without Docker

Create and activate a Python virtual environment, install the required dependencies, and run the Streamlit application.

```powershell
streamlit run .\src\19_streamlit\01_healthai_virtual_hospital.py
```

For the API service:

```powershell
uvicorn 01_fastapi_app:app --app-dir .\src\18_api --host 0.0.0.0 --port 8000
```

---

# Testing

API tests are located under:

```text
test/test_api.py
```

The current test suite covers:

```text
GET  /
GET  /health
POST /predict/diabetes with invalid input
```

---

# Continuous Integration

GitHub Actions configuration is located at:

```text
.github/workflows/ci.yml
```

The workflow is configured to install the API test dependencies and execute the FastAPI test suite in a clean GitHub Actions environment.

---

# Data Governance & Third-Party Resources

HealthAI uses external datasets, pretrained models and public medical-information sources.

Third-party resources remain subject to the terms and licenses provided by their respective original sources.

The datasets and trained model artifacts are intentionally kept outside the Git repository when they are large or externally hosted.

The repository contains the application code, configuration, documentation and reproducible project structure.

---

# Reproducibility

To reproduce the project:

1. Clone the repository.
2. Install the required Python dependencies.
3. Obtain the external datasets from their documented source websites.
4. Restore the required model artifacts.
5. Place the datasets and model artifacts in the expected local directories.
6. Run the FastAPI and Streamlit services.
7. Use the documented source versions and model versions when reproducing experiments.

---

# Project Safety & Intended Use

HealthAI is a portfolio and educational system.

It should not be used as a substitute for:

- Professional medical diagnosis
- Clinical treatment decisions
- Emergency medical services
- Medical interpretation by qualified professionals

Synthetic datasets and demonstration models may not reflect the complexity or variability of real-world clinical environments.

---

# Key Engineering Design Decisions

## Modular adapters

Model-specific preprocessing, inference and output normalization are separated from the UI.

## Stable API contracts

FastAPI provides structured responses between the AI components and Streamlit.

## Agentic routing

Natural-language requests can be routed to specialized healthcare tools.

## Evidence-grounded RAG

Medical-information responses are constructed from retrieved evidence rather than unrestricted generation.

## Safety-aware retrieval

The RAG pipeline uses retrieval quality and a semantic safety threshold before constructing an answer.

## Abstention

The system can abstain when the available retrieval evidence is insufficient.

## Multilingual patient interface

Patient-facing interaction supports English, Hindi and Tamil interface experiences.

## Containerized deployment

Docker packages the application environment and separates large runtime data/model mounts from the application image.

---

# Repository Information

**Repository:**

`HealthAI-Suite-Intelligent-Analytics-for-Patient-Care`

**Primary branch:**

`main`

The repository contains the source code, deployment configuration, documentation and notebooks required to understand the HealthAI implementation.

---

# Attribution

### Dataset sources

**Indian Healthcare Patient Records**  
Kaggle — Arun's Workspace  
https://www.kaggle.com/datasets/arunsworkspace/indian-healthcare-patient-records

**LengthOfStay.csv**  
Microsoft R Server — Hospital Length of Stay  
https://microsoft.github.io/r-server-hospital-length-of-stay/

**Patient Feedback and Sentiment Analysis Dataset**  
Kaggle — SANAK.2000  
https://www.kaggle.com/datasets/sanak2000/cleveland-clinic-patients-feedback

**Hospital Deterioration — Simulated Early Warning**  
Hugging Face — tarekmasryo  
https://huggingface.co/datasets/tarekmasryo/hospital-deterioration-dataset

**Synthetic Chest X-Ray Pneumonia Dataset**  
Hugging Face — chimbiwide  
https://huggingface.co/datasets/chimbiwide/synthetic-chest-xray-pneumonia

**Multilingual Synthetic Medical Notes for NER**  
Hugging Face — E3-JSI  
https://huggingface.co/datasets/E3-JSI/synthetic-multi-med-notes-ner-v1

### Model sources

**BioBERT v1.1**  
https://huggingface.co/dmis-lab/biobert-v1.1

**Bio_ClinicalBERT**  
https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

**DistilBERT base uncased**  
https://huggingface.co/distilbert/distilbert-base-uncased

**all-MiniLM-L6-v2**  
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

**Helsinki-NLP/opus-mt-en-hi**  
https://huggingface.co/Helsinki-NLP/opus-mt-en-hi

---

# Final Note

HealthAI is designed as an end-to-end healthcare AI engineering portfolio demonstrating the integration of:

```text
Machine Learning
      +
Deep Learning
      +
Medical NLP
      +
RAG
      +
Agentic AI
      +
FastAPI
      +
Streamlit
      +
Docker
```

The project emphasizes modular architecture, data provenance, third-party attribution, evidence-grounded medical information, API contracts and explicit safety boundaries.