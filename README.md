# 🚨 Live Email Classification API

An end-to-end machine learning system for detecting spam emails using Gmail-derived email features, Random Forest/XGBoost models, FastAPI, Docker, MLflow, and SHAP explainability.

---

## 🚀 Project Overview

This project evolved from a traditional spam email classifier into a Gmail-oriented machine learning system capable of analyzing real email content and metadata.

The system supports:

- Gmail email feature extraction
- Machine learning based spam detection
- Random Forest and XGBoost models
- Class-balanced training
- MLflow experiment tracking
- SHAP explainability
- FastAPI REST API
- Streamlit dashboard
- Docker deployment
- GitHub Actions CI
- Gmail API authentication

---

## 🧠 Machine Learning Pipeline

```text
                    Gmail Email
                         │
                         ▼
              Email Feature Extraction
                         │
                         ▼
                 Feature Engineering
                         │
                         ▼
                  Feature Scaling
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Random Forest             XGBoost
              │                     │
              └──────────┬──────────┘
                         ▼
                  Model Evaluation
                         │
                         ▼
                  Spam Probability
                         │
                         ▼
                    Prediction
                         │
                         ▼
                SHAP Explainability
````

---

## 🏗️ System Architecture

```text
                         Gmail API
                            │
                            ▼
                  Email Feature Extraction
                            │
                            ▼
                    Feature Engineering
                            │
                            ▼
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
        Random Forest                XGBoost
               │                         │
               └────────────┬────────────┘
                            ▼
                     Model Selection
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
              FastAPI              Streamlit
                 │                     │
                 ▼                     ▼
             REST API          Explainability UI
                 │                     │
                 └──────────┬──────────┘
                            ▼
                          Docker
```

---

## 🔍 Features

### Gmail Integration

The project includes Gmail API integration for working with real email data.

Email information can be processed to generate machine learning features such as:

* Sender information
* Subject characteristics
* Email body characteristics
* Message length
* URL-related features
* HTML-related features
* Suspicious keyword patterns
* Other engineered email metadata

Sensitive Gmail authentication files are kept local and are not committed to GitHub.

---

## 🤖 Machine Learning Models

The project currently compares:

### Random Forest

A tree-based ensemble classifier used as one of the baseline and balanced models.

### XGBoost

A gradient-boosting model used for improved classification performance and comparison against Random Forest.

The training pipeline includes:

* Feature engineering
* Feature scaling
* Class balancing
* Train/test splitting
* Model evaluation
* Experiment tracking
* Model serialization

---

## 📊 Model Evaluation

The project evaluates models using metrics including:

* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

Because spam detection is an imbalanced classification problem, recall and F1 score are considered alongside accuracy.

---

## 🔬 MLflow Experiment Tracking

MLflow is used to track machine learning experiments.

Tracked information can include:

* Model type
* Hyperparameters
* Precision
* Recall
* F1 score
* ROC-AUC
* Model artifacts

This makes it easier to compare Random Forest and XGBoost experiments and reproduce training results.

---

## 🔍 SHAP Explainability

The Streamlit dashboard uses SHAP to explain individual model predictions.

For an analyzed email, the dashboard can show:

* Spam / Legitimate classification
* Spam confidence
* Legitimate confidence
* Top contributing features
* Feature impact
* Raw feature values
* Scaled feature values

Example workflow:

```text
Email
  ↓
Feature Extraction
  ↓
ML Model
  ↓
Prediction
  ↓
SHAP
  ↓
Why was this email classified as spam?
```

---

# ⚡ FastAPI

The project exposes the spam detection model through a FastAPI REST API.

## Health Check

```http
GET /health
```

## Predict Email

```http
POST /predict
```

The API accepts email information and returns a classification and prediction information.

---

## 📖 Interactive API Documentation

When running locally, FastAPI automatically provides Swagger documentation.

Open:

```text
http://localhost:8000/docs
```

This provides an interactive interface for testing the API.

---

# 📊 Streamlit Dashboard

The project also contains an interactive Streamlit dashboard for analyzing individual emails.

The dashboard provides:

* Email subject input
* Sender input
* Email body input
* Spam classification
* Spam confidence
* Legitimate confidence
* SHAP feature explanations
* Debug feature inspection

Run locally with:

```bash
streamlit run src/app_dashboard.py
```

Then open:

```text
http://localhost:8501
```

---

# 🐳 Docker

The FastAPI application can be containerized using Docker.

## Build the image

```bash
docker build -t spam-detector .
```

## Run the API

```bash
docker run --name spam-detector-local -p 8000:8000 spam-detector
```

Then open:

```text
http://localhost:8000/docs
```

---

# 🔐 Gmail API Authentication

The project uses Gmail API authentication for accessing email data.

The following files contain sensitive authentication information and must remain local:

```text
credentials.json
token.pickle
```

These files are excluded through `.gitignore`.

**Never commit Gmail credentials or authentication tokens to GitHub.**

---

# 🔄 CI/CD

GitHub Actions is configured to automatically validate the project when changes are pushed to GitHub or submitted through pull requests.

Current CI checks include:

* Python environment setup
* Dependency installation
* Python syntax validation

Workflow location:

```text
.github/workflows/test.yml
```

---

# 📁 Project Structure

```text
Live-Email-Classification-API/
│
├── .github/
│   └── workflows/
│       └── test.yml
│
├── data/
│
├── model/
│   ├── gmail_feature_names.pkl
│   ├── gmail_random_forest.pkl
│   ├── phase5_feature_names.pkl
│   ├── rf_balanced.pkl
│   ├── scaler.pkl
│   └── xgb_balanced.pkl
│
├── notebooks/
│
├── src/
│   ├── app_dashboard.py
│   ├── build_labeled_dataset.py
│   ├── data_prep.py
│   ├── evaluate.py
│   ├── fetch_gmail_dataset.py
│   ├── gmail_auth.py
│   ├── gmail_features.py
│   ├── predict_gmail_email.py
│   ├── shap_explain.py
│   ├── test_gmail.py
│   ├── train_gmail_model.py
│   ├── train_models.py
│   └── ...
│
├── app.py
├── Dockerfile
├── PHASE0_AUDIT.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Tech Stack

| Category             | Technology     |
| -------------------- | -------------- |
| Programming Language | Python         |
| Machine Learning     | Scikit-learn   |
| Gradient Boosting    | XGBoost        |
| Data Processing      | Pandas, NumPy  |
| Explainability       | SHAP           |
| API                  | FastAPI        |
| Dashboard            | Streamlit      |
| Experiment Tracking  | MLflow         |
| Containerization     | Docker         |
| CI                   | GitHub Actions |
| Email Integration    | Gmail API      |
| Version Control      | Git / GitHub   |

---

# 📌 Project Workflow

The project follows this general workflow:

```text
1. Collect Gmail email data
            ↓
2. Extract email features
            ↓
3. Build labeled dataset
            ↓
4. Train machine learning models
            ↓
5. Compare Random Forest and XGBoost
            ↓
6. Track experiments using MLflow
            ↓
7. Save trained model
            ↓
8. Deploy model through FastAPI
            ↓
9. Explain predictions using SHAP
            ↓
10. Visualize predictions through Streamlit
```

---

# 🚧 Current Status

### Completed

* Gmail-oriented email feature extraction
* Feature engineering
* Random Forest model
* XGBoost model
* Class-balanced training
* MLflow experiment tracking
* SHAP explainability
* FastAPI API
* Streamlit dashboard
* Docker configuration
* GitHub repository
* GitHub Actions CI validation
* Secure handling of Gmail credentials

### Future Improvements

* Public cloud deployment
* Real-time Gmail inbox classification
* Automatic spam labeling
* Email attachment analysis
* Computer vision for image-based spam
* OCR for text contained in email images
* Advanced NLP features
* Model monitoring
* Automated model retraining
* Production-grade authentication
* Cloud-based MLflow tracking

---

# 🌐 Deployment

The project is currently configured for local execution through FastAPI, Streamlit, and Docker.

### Local FastAPI

```text
http://localhost:8000/docs
```

### Local Streamlit Dashboard

```text
http://localhost:8501
```

A public website/live-demo URL will be added here once the application is deployed to a public hosting platform.

> `localhost` URLs are only accessible from the local machine and are not public websites.

---

# 🔒 Security

The repository intentionally excludes sensitive and locally generated files.

Examples:

```text
credentials.json
token.pickle
mlflow.db
mlruns/
data/gmail_dataset.csv
data/gmail_features_dataset.csv
data/gmail_labeled_dataset.csv
```

Before pushing changes to GitHub, always verify:

```bash
git status
```

and make sure no credentials or private Gmail data are staged.

---

# 👩‍💻 Author

## Astha Pankaj

GitHub:

[https://github.com/Asthaasu](https://github.com/Asthaasu)

---

# 📌 Repository

GitHub Repository:

[https://github.com/Asthaasu/Live-Email-Classification-API](https://github.com/Asthaasu/Live-Email-Classification-API)

---

## ⭐ Project Highlights

```text
Gmail API
    +
Feature Engineering
    +
Machine Learning
    +
Random Forest / XGBoost
    +
MLflow
    +
SHAP
    +
FastAPI
    +
Streamlit
    +
Docker
    +
GitHub Actions
```
