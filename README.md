# ChurnGuard AI

Customer Churn Prediction and Retention System using Machine Learning.

## 📌 About the Project

ChurnGuard AI is a machine learning based customer churn prediction system.

The main purpose of this project is to identify customers who are likely to leave a company. The system takes customer information such as tenure, contract type, monthly charges, internet service, payment method and other service details.

Based on this information, the trained machine learning model predicts the probability of customer churn and assigns a risk level.

The system also provides retention recommendations that can help businesses take suitable actions for customers who are at higher risk of leaving.

## 🎯 Project Objectives

- Predict whether a customer is likely to churn.
- Calculate the probability of customer churn.
- Categorize customers into Low, Medium and High risk.
- Understand the factors related to customer churn.
- Provide useful customer retention recommendations.
- Build a simple web interface for making predictions.

## 📊 Dataset

The project uses the Telco Customer Churn dataset.

Dataset information:

- Number of customers: 7,043
- Original number of features: 21
- Target column: `Churn`

The target variable contains:

- `Yes` - Customer churned
- `No` - Customer stayed

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## 🤖 Machine Learning Models

The following classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

The models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Random Forest was further tuned using GridSearchCV.

## 🔄 Machine Learning Workflow

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Preparation
      ↓
Train-Test Split
      ↓
One-Hot Encoding
      ↓
Feature Scaling
      ↓
Model Training
      ↓
Model Comparison
      ↓
Hyperparameter Tuning
      ↓
Final Model
      ↓
Churn Probability
      ↓
Risk Level
      ↓
Retention Recommendation
