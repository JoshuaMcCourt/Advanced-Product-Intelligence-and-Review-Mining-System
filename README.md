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

## ML Pipeline & Intelligence Flow

 ┌──────────────────────────────────────────────────────────────┐
 │                  PRODUCT INTELLIGENCE SYSTEM                  │
 │        Multimodal Review Mining & Fraud Detection            │
 └──────────────────────────────────────────────────────────────┘

                         ┌────────────────────────────┐
                         │       Raw Data Sources     │
                         │                            │
                         │  • Amazon Reviews          │
                         │  • Product Metadata        │
                         │  • (Future) Image Data     │
                         └────────────┬───────────────┘
                                      │
                                      ▼
 ───────────────────────────────────────────────────────────────
                    Data Ingestion Layer
                  (src/data_loader.py)
 ───────────────────────────────────────────────────────────────
   • Schema standardisation
   • ID generation (review_id, product_id)
   • Timestamp parsing & validation
                                      │
                                      ▼
 ───────────────────────────────────────────────────────────────
                    Preprocessing Layer
                  (src/preprocessing.py)
 ───────────────────────────────────────────────────────────────
   • Text cleaning & normalization
   • Duplicate / short review filtering
   • Optional language filtering
   • Structural signals (length, caps, punctuation, z-scores)
                                      │
                                      ▼
 ───────────────────────────────────────────────────────────────
           Feature Engineering & Representation Layer
                    (src/features.py)
 ───────────────────────────────────────────────────────────────
   • Sentence-BERT embeddings (semantic representation)
   • VADER sentiment scoring
   • Behavioural + linguistic features
   • Derived signals (mismatch scores, density, z-scores)
   • Final feature matrix construction (X)
                                      │
                                      ▼
 ───────────────────────────────────────────────────────────────
                       Modeling Layer
                     (src/models.py)
 ───────────────────────────────────────────────────────────────
   • Isolation Forest → anomaly detection
   • KMeans → semantic clustering
   • Cluster assignment + distance-to-centroid
   • Feature scaling (structured + embeddings)
   • Risk feature matrix construction
                                      │
                                      ▼
 ───────────────────────────────────────────────────────────────
                   Risk Intelligence Layer
          (src/models.py + src/calibration.py)
 ───────────────────────────────────────────────────────────────
   • Heuristic risk score (multi-signal fusion)
   • Risk normalization (bounded [0,1])
   • Logistic regression calibration model
   • Calibrated fraud probability output
                                      │
                                      ▼
 ───────────────────────────────────────────────────────────────
                 Temporal Intelligence Layer
               (src/temporal_analysis.py)
 ───────────────────────────────────────────────────────────────
   • Review velocity (volume trends)
   • Sentiment drift detection
   • Fraud spike detection (z-score anomalies)
   • Multi-window aggregation (1D / 7D / 30D)
                                      │
                                      ▼
 ───────────────────────────────────────────────────────────────
                   Explainability Engine
               (src/explainability.py)
 ───────────────────────────────────────────────────────────────
   • Review-level explanations (why flagged)
   • Risk factor attribution
   • Cluster-level summaries (top words + behaviour)
   • Audit-ready interpretability outputs
                                      │
                                      ▼
 ───────────────────────────────────────────────────────────────
                     Evaluation Layer
                 (src/evaluation.py)
 ───────────────────────────────────────────────────────────────
   • Anomaly detection metrics (ROC-AUC, PR-AUC)
   • Clustering quality (silhouette, entropy)
   • Distribution diagnostics & correlations
   • Product-level scoring & validation
                                      │
                                      ▼
 ───────────────────────────────────────────────────────────────
                   Visualization Layer
               (src/visualization.py)
 ───────────────────────────────────────────────────────────────
   • Risk score distributions
   • Cluster distributions
   • Static chart generation → outputs/charts/
                                      │
                                      ▼
 ───────────────────────────────────────────────────────────────
               Output & Persistence Layer
 ───────────────────────────────────────────────────────────────
   • Processed dataset → data/processed/processed_reviews.csv
   • Models → models/
       - isolation_forest.pkl
       - kmeans.pkl
       - calibration_model.pkl
   • Scalers → models/
       - scaler.pkl
       - scaler_structured.pkl
       - scaler_embeddings.pkl
       - scaler_calibration.pkl
   • Reports → outputs/reports/
       - evaluation.json
       - temporal_analysis.json
       - cluster_explanations.json
       - review_explanations.csv
       - feature_columns.json
   • Charts → outputs/charts/
                                      │
                                      ▼
 ───────────────────────────────────────────────────────────────
                 Interface & Delivery Layer
 ───────────────────────────────────────────────────────────────
   • run_pipeline.py → config-driven execution entry point
   • notebooks/full_notebook.ipynb → end-to-end narrative
   • app/dashboard.py → Streamlit dashboard
                                      │
                                      ▼
 ───────────────────────────────────────────────────────────────
                Product Intelligence Outputs
 ───────────────────────────────────────────────────────────────
   • Fraud probability per review
   • Risk scores + anomaly flags
   • Product health indicators
   • Cluster-based behavioural segmentation
   • Temporal fraud insights (spikes & drift)
   • Fully explainable AI decisions


## Folder Structure & Detailed Contents

product-intelligence-system/
│
├── data/                          # Raw and processed datasets
│   ├── raw/                       # Original input data
│   │   ├── amazon_reviews.csv
│   │   └── electronics_products.csv
│   │
│   └── processed/                 # Pipeline outputs (model-ready dataset)
│       └── processed_reviews.csv
│
├── notebooks/                     # Exploratory + narrative notebooks
│   └── full_notebook.ipynb        # End-to-end walkthrough (portfolio-facing)
│
├── src/                           # Core ML pipeline modules
│   │
│   ├── __init__.py
│   ├── utils.py                   # Logging, timers, IO utilities, persistence
│   │
│   ├── data_loader.py             # Data ingestion + schema standardisation
│   ├── preprocessing.py           # Text cleaning + normalization
│   ├── features.py                # Feature engineering (NLP + embeddings)
│   ├── models.py                  # Anomaly detection + clustering + risk scoring
│   ├── calibration.py             # Probability calibration (risk → fraud likelihood)
│   ├── explainability.py          # Review + cluster explainability
│   ├── temporal_analysis.py       # Time-series intelligence + anomaly detection
│   ├── evaluation.py              # Full evaluation + diagnostics suite
│   ├── visualization.py           # Static chart generation
│   │
│   └── pipeline.py                # End-to-end orchestration logic
│
├── run_pipeline.py                # Execution entry point (config-driven pipeline)
│
├── models/                        # Serialized trained models
│   ├── isolation_forest.pkl       # Anomaly detection model
│   ├── kmeans.pkl                 # Clustering model
│   ├── calibration_model.pkl      # Probability calibration model
│   ├── scaler.pkl                 # General feature scaler
│   ├── scaler_structured.pkl      # Structured feature scaler
│   ├── scaler_embeddings.pkl      # Embedding scaler
│   └── scaler_calibration.pkl     # Calibration scaler
│
├── outputs/
│   ├── charts/                   # Generated visualisations
│   │   ├── risk_distribution.png
│   │   └── cluster_distribution.png
│   │
│   └── reports/                  # Analytical outputs and diagnostics
│       ├── evaluation.json            # Model performance + system diagnostics
│       ├── temporal_analysis.json     # Time-based trends and anomaly signals
│       ├── cluster_explanations.json  # Cluster-level interpretability
│       ├── review_explanations.csv    # High-risk review explanations
│       └── feature_columns.json       # Feature schema for reproducibility
│
├── app/
│   └── dashboard.py              # Streamlit dashboard for exploration
│
├── config.yaml                   # Central configuration (paths, hyperparameters)
├── requirements.txt              # Dependencies
├── .gitignore                    # Exclusions (data, models, outputs)
└── README.md                     # Project documentation
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
