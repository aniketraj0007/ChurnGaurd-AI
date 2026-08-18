# ChurnGuard AI

Customer Churn Prediction & Retention System built using Python, Scikit-Learn, and Streamlit.

## About the Project
ChurnGuard AI is a machine learning project that predicts whether a customer is likely to cancel their service (churn). The project uses customer attributes such as tenure, contract type, monthly charges, and subscribed services to estimate churn probability. I also added a simple retention recommendation system that suggests possible actions for high-risk customers.

## Problem Statement
Customer churn is a critical problem for subscription-based businesses like telecommunications. Acquiring new customers is generally more expensive than retaining existing ones. Predicting churn early allows customer success teams to take proactive retention steps before a subscriber cancels.

## Dataset
The project uses the publicly available **Telco Customer Churn** dataset (7,043 rows, 21 columns). 
Key features include:
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Account Services**: Phone Service, Internet Service, Tech Support, Online Security, etc.
- **Contract & Billing**: Contract Type (Month-to-month, 1-Year, 2-Year), Payment Method, Monthly Charges, Total Charges
- **Target**: `Churn` (Yes / No)

## What I Did
1. **Data Cleaning & Preprocessing**: Handled blank/missing values in `TotalCharges`, converted categorical variables using One-Hot Encoding, and scaled numerical features with StandardScaler.
2. **Exploratory Data Analysis (EDA)**: Analyzed churn rate distributions across contract types, tenure buckets, and monthly charges.
3. **Model Training & Hyperparameter Tuning**: Evaluated 5 machine learning algorithms using 5-fold cross-validation and stratified train-test splits.
4. **Retention Engine**: Created a rule-based recommendation logic to generate specific retention actions based on customer risk factors.
5. **Interactive Web Dashboard**: Built an interactive Streamlit application to visualize churn metrics, score individual customer risk, view high-risk accounts, and compare model performance.

## Machine Learning Models
I trained and evaluated the following classification algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- CatBoost

## Results
The models were evaluated on a 20% test dataset using Accuracy, Precision, Recall, F1 Score, and ROC-AUC:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CatBoost** | 0.8119 | 0.6728 | 0.5481 | 0.6041 | **0.8492** |
| **XGBoost** | 0.8041 | 0.6559 | 0.5401 | 0.5924 | 0.8451 |
| **Logistic Regression** | 0.8027 | 0.6472 | 0.5535 | 0.5967 | 0.8443 |
| **Random Forest** | 0.8020 | 0.6548 | 0.5294 | 0.5855 | 0.8423 |
| **Decision Tree** | 0.7878 | 0.6120 | 0.5481 | 0.5783 | 0.8091 |

*Note: CatBoost / XGBoost achieved the best overall ROC-AUC score (~0.849).*

## Features
- **Interactive Dashboard**: Top KPI metrics and key churn analytics charts.
- **Customer Churn Form**: Real-time churn prediction, probability scoring, risk level assignment, and top risk factors.
- **Retention Center**: Actionable table of high-risk customers with filter options.
- **Model Performance**: Educational comparison of classification models with interactive metrics bar charts.
- **Customer Analysis**: Detailed exploratory breakdown of churn drivers.

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Model
```bash
python run.py --train
```

### 3. Launch the Streamlit Web Application
```bash
python run.py --app
```
Access the application in your browser at `http://localhost:8501`.

## Project Structure
```
ChurnGuard AI/
│
├── data/
│   └── telco_customer_churn.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
│
├── models/
│   ├── churn_model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.json
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── retention.py
│
├── app/
│   ├── app.py
│   └── pages/
│       ├── 1_Dashboard.py
│       ├── 2_Churn_Prediction.py
│       ├── 3_Retention.py
│       ├── 4_Model_Performance.py
│       └── 5_Customer_Analysis.py
│
├── run.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Future Improvements
- Test hyperparameter tuning with Optuna.
- Collect more historical customer interaction data to improve model recall.
- Experiment with SMOTE or class weighting for improved minority class detection.
