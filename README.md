HealthAI — Intelligent Healthcare AI Platform

An end-to-end healthcare AI portfolio platform integrating classical machine learning, deep learning, medical NLP, retrieval-augmented generation (RAG), agentic orchestration, multilingual patient interaction, FastAPI, Streamlit, and Docker.

Important: HealthAI is an educational/portfolio demonstration. It is not a medical device and is not intended for diagnosis, treatment decisions, or clinical decision-making.

Overview

HealthAI brings multiple healthcare AI workflows into one modular application:

Disease-risk / healthcare classification

Hospital length-of-stay regression

Primary diagnosis prediction

Patient clustering

Association-rule mining

CNN chest X-ray classification

RNN/LSTM patient deterioration workflow

Patient feedback sentiment analysis

Medical named-entity recognition (NER)

Evidence-grounded medical RAG

Agentic routing and tool execution

English, Hindi, and Tamil patient-facing interface

English-to-Hindi neural machine translation

Architecture

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

Project Structure

HealthAI-Suite-Intelligent-Analytics-for-Patient-Care/
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements-runtime.txt
│
├── docs/
│   └── classification/
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

AI / ML Modules

Module

Main approach

Purpose

Healthcare classification

Classical ML

Structured healthcare risk / category prediction

Length of stay

Regression / XGBoost

Predict hospital stay duration

Primary diagnosis

Classification

Predict configured diagnosis categories

Patient clustering

Unsupervised learning

Segment patient records into data-driven groups

Association mining

Apriori / association rules

Discover recurring feature relationships

Chest X-ray

CNN

Normal / pneumonia image classification

Deterioration

RNN / LSTM

Sequential patient deterioration modelling

Sentiment

Transformer / DistilBERT workflow

Classify patient feedback sentiment

Medical NER

BioBERT / ClinicalBERT workflow

Extract medical entities from text

Medical RAG

Retrieval + embeddings

Ground medical-information responses in curated sources

Agentic AI

Router + executor + adapters

Route user requests to specialized healthcare tools

Translation

Helsinki-NLP NMT

English-to-Hindi translation

Data Provenance & Dataset Download Sources

The following records list the external datasets used by the project and their source websites.

Indian Healthcare Patient Records

Use: Main tabular healthcare modelling, classification, clustering and association workflows
Source: Kaggle — Arun's Workspace

LengthOfStay.csv

Use: Hospital length-of-stay regression and ANN/MLP workflows
Source: Microsoft R Server — Hospital Length of Stay
License: MIT

Patient Feedback and Sentiment Analysis Dataset

Use: Patient feedback, sentiment and satisfaction analysis
Source: Kaggle — Patient Feedback and Sentiment Analysis Dataset
License: MIT
Local file: data/raw/medical_sentiment/patient_feedback_dataset.xlsx

Healthcare_DataSet.csv

Use: Medical-text / sentiment exploration
Local file: data/raw/medical_sentiment/Healthcare_DataSet.csv
Records: 87,057

Hospital Deterioration — Simulated Early Warning

Use: RNN/LSTM patient deterioration workflow
Source: Hugging Face — tarekmasryo/hospital-deterioration-dataset
License: CC BY 4.0

Synthetic Chest X-Ray Pneumonia Dataset

Use: CNN normal/pneumonia image classification
Source: Hugging Face — chimbiwide/synthetic-chest-xray-pneumonia
License: CC BY 4.0

Multilingual Synthetic Medical Notes for NER

Use: Medical named-entity recognition
Source: Hugging Face — E3-JSI/synthetic-multi-med-notes-ner-v1
License: MIT

Dataset status in Git

The external datasets and generated data are excluded from the Git repository. The project .gitignore excludes:

models/
mlruns/
data/raw/
data/processed/

This keeps the downloaded source data outside Git history. Re-download or restore the datasets from their documented source websites when reproducing the project.

Pretrained Model Sources

BioBERT v1.1

Use: Biomedical NER
Source: Hugging Face — dmis-lab/biobert-v1.1

Bio_ClinicalBERT

Use: Clinical NLP / NER
Source: Hugging Face — emilyalsentzer/Bio_ClinicalBERT
License: MIT

DistilBERT base uncased

Use: Sentiment transformer workflow
Source: Hugging Face — distilbert/distilbert-base-uncased
License: Apache-2.0

all-MiniLM-L6-v2

Use: RAG embeddings
Source: Hugging Face — sentence-transformers/all-MiniLM-L6-v2
License: Apache-2.0

Helsinki-NLP/opus-mt-en-hi

Use: English-to-Hindi translation
Source: Hugging Face — Helsinki-NLP/opus-mt-en-hi
License: Apache-2.0

Medical RAG Sources

The RAG source registry contains eight curated public medical-information sources. The exact URLs are retained in the project source registry.

Organization

Topic

Source website

World Health Organization

Diabetes

https://www.who.int/health-topics/diabetes

WHO India

Diabetes

https://www.who.int/india/health-topics/diabetes

World Health Organization

Hypertension

https://www.who.int/health-topics/hypertension/

WHO India

Hypertension

https://www.who.int/india/health-topics/hypertension

World Health Organization

Noncommunicable Diseases

https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases

MedlinePlus / U.S. National Library of Medicine

General Health

https://medlineplus.gov/healthtopics.html

MedlinePlus / U.S. National Library of Medicine

Type 2 Diabetes Self-Care

https://medlineplus.gov/ency/patientinstructions/000328.htm

MedlinePlus / U.S. National Library of Medicine

Living With Chronic Illness

https://medlineplus.gov/ency/patientinstructions/000602.htm

These are knowledge sources, not datasets owned by this project. Their individual terms and copyright conditions continue to apply.

Translation

The medical translation worker uses:

Helsinki-NLP/opus-mt-en-hi

This is a pretrained neural machine-translation model for English-to-Hindi translation. It is not presented as a general-purpose LLM.

X-Ray Safety / Intended Use

The chest X-ray module is explicitly positioned as a portfolio demonstration using a synthetic dataset. It is not intended for clinical diagnosis or medical decision-making.

Docker Deployment

The project includes a Docker deployment under deployment/.

Start the application

docker compose -f .\deployment\docker-compose.yml up --build

Application URLs

Streamlit: http://localhost:8501
FastAPI:   http://localhost:8000

The deployment expects large datasets and model artifacts to be available locally because data/ and models/ are excluded from the Docker build context and mounted at runtime by the Compose configuration.

API

FastAPI provides the service layer between the Streamlit interface and the healthcare AI tools.

Representative endpoints include:

GET  /
GET  /health
POST /predict/diabetes
POST /predict/xray
POST /predict/los
POST /predict/diagnosis

API responses are normalized so the presentation layer does not need to depend on every model's internal output structure.

Testing / CI

The repository contains API tests under test/test_api.py and a GitHub Actions workflow under .github/workflows/ci.yml.

The CI workflow runs the API tests in a clean Python environment and installs the required test dependencies, including python-multipart for FastAPI file/form endpoints.

Reproducibility Notes

Keep the externally downloaded datasets in the locations expected by the project code.

Keep model artifacts outside Git and restore/download them from their documented sources.

Use the same source dataset/version when reproducing model-training experiments.

Do not commit secrets, .env files, virtual environments, model artifacts, MLflow runs, or large raw datasets.

Attribution & References

Core dataset references

Indian Healthcare Patient Records: https://www.kaggle.com/datasets/arunsworkspace/indian-healthcare-patient-records

Length of Stay data: https://microsoft.github.io/r-server-hospital-length-of-stay/

Patient Feedback and Sentiment Analysis Dataset: https://www.kaggle.com/datasets/sanak2000/cleveland-clinic-patients-feedback

Hospital Deterioration dataset: https://huggingface.co/datasets/tarekmasryo/hospital-deterioration-dataset

Synthetic Chest X-Ray Pneumonia dataset: https://huggingface.co/datasets/chimbiwide/synthetic-chest-xray-pneumonia

Multilingual Synthetic Medical Notes for NER: https://huggingface.co/datasets/E3-JSI/synthetic-multi-med-notes-ner-v1

Model references

BioBERT: https://huggingface.co/dmis-lab/biobert-v1.1

Bio_ClinicalBERT: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

DistilBERT: https://huggingface.co/distilbert/distilbert-base-uncased

all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Helsinki-NLP English-to-Hindi NMT: https://huggingface.co/Helsinki-NLP/opus-mt-en-hi

RAG source references

The exact WHO and MedlinePlus URLs used by the RAG source registry are listed above in Medical RAG Sources.

Limitations

The project uses public and/or synthetic resources rather than private hospital records.

Model outputs are portfolio demonstrations and are not clinically validated.

Dataset quality, representativeness, licensing, and external-source terms remain specific to each original source.

External datasets and model artifacts are intentionally excluded from the Git repository.