# BehaviorGuard — Customer Behavioral Profiling for Credit Card Fraud Detection

## Run the API

From the repository root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cd "BehaviorGuard — Customer Behavioral Profiling for Credit Card Fraud Detection"
uvicorn api.main:app --reload
```

The service is available at `http://127.0.0.1:8000`. Interactive API documentation is available at `/docs`.

The API loads the existing XGBoost pipeline once at startup. Configure local frontend origins with `BEHAVIORGUARD_CORS_ORIGINS` as a comma-separated list when needed.

> An ML-powered fraud detection system that learns customer spending behavior, identifies unusual transaction patterns, and generates explainable fraud risk scores.

---

## Overview

**BehaviorGuard** is a machine learning-based credit card fraud detection system focused on **customer behavioral profiling**.

Instead of treating every transaction independently, the system learns a customer's historical spending patterns and compares each new transaction against their established behavioral profile. Transactions that significantly deviate from normal behavior are assigned a higher fraud risk.

The project combines:

- Customer behavioral profiling
- Feature engineering
- Anomaly detection
- Supervised machine learning
- Fraud risk scoring
- Model evaluation
- Explainable predictions

### Core Concept

```text
Customer Transaction History
            │
            ▼
   Behavioral Profiling
            │
            ▼
   Behavioral Features
            │
            ▼
    Fraud Detection Model
            │
            ▼
    Fraud Probability
            │
            ▼
     Risk Assessment
            │
      ┌─────┴─────┐
      ▼           ▼
    Low Risk    High Risk
    /Normal      /Alert
```

The goal is to build a system that can answer not only:

> **"Is this transaction fraudulent?"**

but also:

> **"How unusual is this transaction for this particular customer, and why was it flagged?"**

---

##  Features

### Customer Behavioral Profiling

Creates a behavioral baseline for each customer using historical transaction information such as:

- Average transaction amount
- Median transaction amount
- Maximum transaction amount
- Transaction frequency
- Typical transaction time
- Spending patterns
- Merchant/category preferences
- Historical transaction statistics

### Behavioral Anomaly Detection

Identifies transactions that deviate from a customer's normal behavior.

Examples include:

- Unusually large transactions
- Unusual transaction times
- Abnormally high transaction frequency
- Unexpected merchant categories
- Significant changes in spending patterns

### Machine Learning Fraud Detection

Supports multiple machine learning approaches for comparison, including:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Isolation Forest

### Fraud Risk Scoring

Instead of producing only a binary prediction, the system can generate a fraud probability or risk score.

Example:

```text
Fraud Probability: 91%
Risk Level: HIGH
```

### Explainable Fraud Detection

Provides interpretable reasons for suspicious transactions, such as:

```text
⚠ Transaction amount is significantly higher than customer's average
⚠ Transaction occurred outside the customer's normal transaction hours
⚠ Transaction frequency is unusually high
```

### Model Evaluation

Evaluates models using fraud-appropriate metrics such as:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

### API Integration

The trained model can be exposed through a REST API for real-time or near-real-time transaction scoring.

---

# Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Anomaly Detection | Isolation Forest |
| Explainability | SHAP |
| Backend API | FastAPI |
| Database | PostgreSQL |
| Frontend | React |
| Version Control | Git & GitHub |
| Deployment | Docker / Cloud Platform |

---

# Project Structure

```text
BehaviorGuard/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_customer_profiling.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── profiling.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── evaluation.py
│
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── api/
│   └── main.py
│
├── frontend/
│
├── tests/
│
├── requirements.txt
├── .gitignore
├── Dockerfile
└── README.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/BehaviorGuard.git
cd BehaviorGuard
```

Replace `your-username` with the GitHub username that owns the repository.

---

## 2. Create a Virtual Environment

It is recommended to use a virtual environment to isolate the project's dependencies.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not yet been created, the main dependencies include:

```text
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
plotly
shap
fastapi
uvicorn
joblib
```

---

## 4.the Dataset


```text
data/
└── raw/
    └── the Kaggle Credit Card Fraud Detection dataset
```


---

## 5. Run the Notebooks

Start Jupyter:

```bash
jupyter notebook
```

Then execute the notebooks in the recommended order:

```text
01_eda.ipynb
        ↓
02_data_preprocessing.ipynb
        ↓
03_customer_profiling.ipynb
        ↓
04_feature_engineering.ipynb
        ↓
05_model_training.ipynb
        ↓
06_model_evaluation.ipynb
```

---

#  Usage

## Data Processing

The preprocessing pipeline cleans and transforms raw transaction data before model training.

Typical operations include:

```python
import pandas as pd

df = pd.read_csv("data/raw/transactions.csv")

df = df.drop_duplicates()
df = df.dropna()

print(df.shape)
```

The actual preprocessing steps depend on the dataset being used.

---

## Customer Profiling

A customer's historical transactions can be aggregated to create a behavioral profile.

Example profile:

```text
Customer ID: 1024

Average Transaction: $78.50
Maximum Transaction: $240.00
Transactions/Day: 4
Typical Transaction Time: 18:00–22:00
```

This profile is then used as a baseline for evaluating future transactions.

---

## Fraud Prediction

After training the model, a new transaction can be passed through the prediction pipeline.

Example:

```python
transaction = {
    "customer_id": 1024,
    "amount": 850.50,
    "merchant_category": "jewelry",
    "transaction_hour": 3
}
```

Example prediction:

```text
Fraud Probability: 0.91
Behavioral Anomaly Score: 0.87
Risk Level: HIGH
```

---

# API Usage

BehaviorGuard can expose the fraud detection model through a FastAPI service.

Start the API with:

```bash
uvicorn api.main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive documentation can be accessed through:

```text
http://127.0.0.1:8000/docs
```

### Example Request

```http
POST /predict
```

```json
{
  "customer_id": 1024,
  "amount": 850.50,
  "merchant_category": "jewelry",
  "transaction_hour": 3
}
```

### Example Response

```json
{
  "fraud_probability": 0.91,
  "behavioral_anomaly_score": 0.87,
  "risk_level": "HIGH",
  "reasons": [
    "Unusually high transaction amount",
    "Unusual transaction time",
    "Unusual merchant category"
  ]
}
```

---

# Dataset

## Dataset Overview

The project is designed for transaction-level credit card fraud datasets containing historical transaction information and, ideally, customer identifiers.

A suitable dataset may contain fields such as:

```text
customer_id
transaction_id
timestamp
amount
merchant
merchant_category
location
payment_method
fraud_label
```

The exact schema depends on the selected dataset.

---

## Dataset Source

For the initial implementation, a publicly available credit card fraud dataset can be used.

A commonly used benchmark is the **Credit Card Fraud Detection** dataset available through Kaggle and other public machine learning repositories.

> Dataset source should be explicitly credited in the project documentation according to the license and attribution requirements of the specific dataset selected.

If a dataset containing `customer_id` and richer transaction metadata is selected, it is preferable for this project because customer-level behavioral profiling requires historical transactions to be associated with individual customers.

---

## Dataset Format

The expected format is generally:

```text
CSV
```

Example:

```text
data/raw/transactions.csv
```

---

## Preprocessing

Typical preprocessing steps include:

1. Removing duplicate records
2. Handling missing values
3. Converting timestamps into usable temporal features
4. Encoding categorical variables
5. Scaling numerical features where appropriate
6. Sorting transactions chronologically
7. Creating customer-level historical statistics
8. Generating behavioral deviation features
9. Handling class imbalance
10. Splitting data into training, validation, and test sets

### Important: Prevent Data Leakage

Behavioral features must only use information available **before the transaction being predicted**.

For example:

```text
Previous Transactions
        ↓
Customer Profile
        ↓
Current Transaction
        ↓
Fraud Prediction
```

Future transactions must not be used when constructing the historical profile for the current transaction.

This is particularly important for realistic fraud detection evaluation.

---

# Machine Learning Approach

BehaviorGuard follows a multi-stage approach.

## 1. Baseline Classification

Initial models can include:

```text
Logistic Regression
Decision Tree
Random Forest
```

## 2. Advanced Classification

The project can then evaluate:

```text
XGBoost
```

for improved performance on tabular transaction data.

## 3. Anomaly Detection

An unsupervised model such as:

```text
Isolation Forest
```

can be used to identify unusual transaction behavior.

## 4. Behavioral + ML Hybrid Model

The final system can combine:

```text
Behavioral Anomaly Score
            +
ML Fraud Probability
            ↓
     Final Risk Score
```

This allows the system to consider both:

- General fraud patterns learned from the dataset
- Individual customer behavioral deviations

---

#  Evaluation

Because credit card fraud datasets are typically highly imbalanced, accuracy alone is not an appropriate measure of model performance.

BehaviorGuard focuses on:

### Precision

Measures how many transactions predicted as fraud are actually fraudulent.

### Recall

Measures how many actual fraudulent transactions are successfully detected.

### F1-Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between fraudulent and legitimate transactions across different thresholds.

### PR-AUC

Particularly useful for highly imbalanced fraud detection problems.

### Confusion Matrix

Used to analyze:

```text
True Positives
True Negatives
False Positives
False Negatives
```

---

# Explainability

Fraud detection systems should ideally provide reasons behind their predictions.

BehaviorGuard can use behavioral deviations and explainability techniques such as SHAP to identify important factors.

Example:

```text
Transaction Risk: HIGH

Primary Factors:

1. Transaction amount significantly above customer's normal range
2. Transaction occurred during an unusual hour
3. Transaction frequency exceeded historical behavior
4. Merchant category differs from customer's typical spending pattern
```

This makes the system easier for fraud analysts to interpret.

---

# Class Imbalance

Fraud datasets usually contain significantly fewer fraudulent transactions than legitimate transactions.

Potential approaches include:

- Class weighting
- Random under-sampling
- Random over-sampling
- SMOTE
- Threshold tuning
- Ensemble methods

The selected technique should be evaluated carefully because inappropriate resampling can introduce bias or data leakage.

---

# Contributing

Contributions are welcome.

To contribute:

### 1. Fork the repository

Create your own fork of the project.

### 2. Create a feature branch

```bash
git checkout -b feature/your-feature
```

### 3. Make your changes

Follow the existing project structure and coding conventions.

### 4. Run tests

```bash
pytest
```

### 5. Commit your changes

```bash
git add .
git commit -m "Add your feature"
```

### 6. Push the branch

```bash
git push origin feature/your-feature
```

### 7. Open a Pull Request

Describe:

- What was changed
- Why it was changed
- How it was tested
- Any relevant limitations

---

# Security & Privacy

Credit card transaction data can contain sensitive financial information.

When working with real-world data:

- Never commit card numbers to the repository.
- Never expose personally identifiable information.
- Never commit API keys or passwords.
- Use environment variables for secrets.
- Use anonymized or publicly licensed datasets for development.
- Follow applicable data-protection and financial-data regulations.

The project is intended for **educational, research, and prototyping purposes** unless appropriately validated and adapted for production use.

---

# License

This project is released under the **MIT License**.

You may use, modify, and distribute the project according to the terms of the license.

See the [`LICENSE`](LICENSE) file for the complete license text.

> Note: The project code license does not automatically grant rights to redistribute the dataset. Dataset usage must comply with the dataset provider's own terms and license.

---

#  Contact

**Project:** BehaviorGuard  
**Purpose:** Customer Behavioral Profiling for Credit Card Fraud Detection

For questions, suggestions, bug reports, or collaboration opportunities, please open an **Issue** in this repository or contact the project maintainer through the contact information associated with the GitHub repository.

---

#  Future Improvements

Potential future extensions include:

- Real-time transaction monitoring
- Online/incremental learning
- Customer clustering and segmentation
- Graph-based fraud detection
- Deep learning models
- Graph Neural Networks for transaction relationships
- Advanced SHAP-based explanations
- Real-time alert generation
- Email/SMS fraud alerts
- React-based monitoring dashboard
- PostgreSQL transaction history
- Dockerized deployment
- Cloud deployment
- Model monitoring and drift detection

---

# Disclaimer

BehaviorGuard is an educational and research-oriented project.

A machine learning prediction should not be considered definitive proof of fraudulent activity. In a real financial environment, fraud decisions should incorporate additional security controls, human review, regulatory requirements, and appropriately validated production systems.

---

## Project Goal

The ultimate goal of BehaviorGuard is to move beyond conventional transaction-level fraud classification toward **personalized fraud detection based on individual customer behavior**.

```text
Normal Customer Behavior
          ↓
Customer Behavioral Profile
          ↓
Behavioral Deviation
          ↓
Machine Learning
          ↓
Fraud Probability
          ↓
Explainable Risk Assessment
```

**BehaviorGuard — Detect fraud by understanding behavior.**