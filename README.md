# Product Intelligence System (NLP | Machine Learning | Explainable AI)

## Overview

This project implements a complete end-to-end machine learning pipeline for analysing product reviews at scale, with a focus on fraud detection, behavioural intelligence, and explainable insights.

It is designed to replicate a **production-grade ML system**, combining natural language processing, anomaly detection, clustering, and temporal analytics into a unified modular architecture.

The system transforms raw review data into structured intelligence that can be used for monitoring trust, detecting manipulation, and understanding product performance.

---

## Business Context & Value

In modern e-commerce platforms, user-generated reviews directly influence purchasing decisions, product rankings, and platform trust.

However, review systems are vulnerable to:

- Fake or manipulated reviews
- Coordinated spam campaigns
- Sentiment distortion
- Behavioural anomalies

This project demonstrates how machine learning can be applied to:

- Detect fraudulent or suspicious reviews
- Monitor product trust and quality signals
- Identify behavioural patterns across users and products
- Support risk-based decision making
- Provide interpretable insights for moderation systems

---

## Key Analytical Capabilities

The system enables deep analysis across textual, behavioural, and temporal dimensions:

- Detection of **anomalous reviews using Isolation Forest**
- Identification of **semantic clusters of reviews using embeddings**
- Measurement of **sentiment vs rating inconsistencies**
- Detection of **outliers in linguistic and behavioural patterns**
- Generation of **fraud probability scores via calibration models**
- Analysis of **temporal spikes in review activity and fraud signals**
- Extraction of **cluster-level themes and insights**
- Product-level intelligence including **risk and quality scoring**

---

## Pipeline Architecture

The system follows a structured ML pipeline similar to real-world production systems:

Raw Review Data

   ↓

Data Ingestion & Schema Standardisation

   ↓

Text Preprocessing & Cleaning

   ↓

Feature Engineering (NLP + Behavioural + Statistical)

   ↓

Modeling (Anomaly Detection + Clustering)

   ↓

Risk Scoring System

   ↓

Calibration (Probability Estimation)

   ↓

Explainability Layer

   ↓

Temporal Analysis

   ↓

Evaluation & Diagnostics

   ↓

Visualisation & Dashboard

---

## System Design

The architecture is modular and designed for extensibility and reproducibility:

- **data_loader** – schema standardisation, ID generation, timestamp handling 
- **preprocessing** – text cleaning, filtering, linguistic feature extraction 
- **features** – sentiment analysis, embeddings, feature matrix construction  
- **models** – anomaly detection, clustering, risk signal generation  
- **calibration** – conversion of risk signals into calibrated probabilities  
- **explainability** – human-readable explanations for reviews and clusters  
- **temporal_analysis** – time-based anomaly detection and drift analysis  
- **evaluation** – full ML diagnostics and performance analysis  
- **visualization** – generation of charts for reporting and dashboards  
- **pipeline** – orchestration of the full system  

---

## Data Model & Feature Design

The system constructs a rich feature space combining:

### Textual Features
- Cleaned review text
- Sentence embeddings (MiniLM)
- Sentiment scores (VADER)

### Behavioural Features
- Review length and word count
- Capitalisation patterns
- Punctuation density
- Exclamation usage

### Derived Features
- Sentiment-rating gap
- Length z-scores
- Word density metrics
- Cluster distance (semantic deviation)

### Risk Features
- Anomaly scores
- Behavioural inconsistencies
- Semantic outliers
- Linguistic anomalies

These features are fused into a **unified feature matrix** used for modeling and risk scoring.

---

## What This Project Demonstrates

### Machine Learning Engineering
- End-to-end pipeline design with modular components
- Integration of multiple ML models in a single system
- Feature fusion across structured and unstructured data
- Robust handling of real-world noisy datasets

### NLP & Representation Learning
- Sentiment analysis using VADER
- Semantic embeddings using Sentence Transformers
- Text preprocessing and linguistic feature extraction

### Anomaly Detection & Clustering
- Isolation Forest for unsupervised fraud detection
- KMeans clustering for semantic segmentation
- Distance-based outlier detection

### Explainable AI
- Review-level reasoning for flagged content
- Cluster-level interpretability (themes + behaviour)
- Transparent risk scoring logic

### Temporal Intelligence
- Detection of spikes in fraudulent activity
- Sentiment drift analysis over time
- Review velocity monitoring

### System Design
- Clean modular architecture
- Reproducibility via configuration
- Separation of pipeline stages
- Production-style code organisation

---

## Scale & Performance

- Designed to handle large-scale review datasets
- Efficient feature computation and vectorised operations
- Embedding generation with batch processing
- Scalable architecture for extension to larger datasets
- Optimised for both experimentation and production-style workflows

---

## ML Engineering Highlights

- Feature fusion of structured + embedding vectors
- Stable risk scoring system with bounded outputs
- Calibration using logistic regression for probability estimation
- Separation of structured vs embedding scaling pipelines
- Robust handling of missing and degenerate data cases
- Defensive programming (schema checks, assertions, safeguards)

---

## Outputs

### Charts
- Risk score distribution
- Cluster distribution

### Reports

- `evaluation.json` – Model performance and diagnostics  
- `temporal_analysis.json` – Time-based insights (drift, spikes, trends)  
- `cluster_explanations.json` – Cluster-level summaries and themes  
- `review_explanations.csv` – Row-level explainability outputs  
- `feature_columns.json` – Feature schema for reproducibility  

---

## Dashboard

An interactive Streamlit dashboard is included for exploration:

Features:
- Risk filtering and exploration
- Cluster visualisation
- Sentiment vs risk analysis
- Temporal trend monitoring
- High-risk review inspection
- Product-level intelligence view

Run with:

```bash
streamlit run app/dashboard.py
