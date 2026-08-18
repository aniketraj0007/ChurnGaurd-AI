import pandas as pd
import numpy as np
from src.model import load_model
from src.retention import get_risk_level, get_retention_action


def predict_churn(customer_dict):
    """
    Predicts churn probability for a single customer.
    """
    model, preprocessor, _ = load_model()
    if model is None or preprocessor is None:
        raise ValueError("Model or preprocessor not found. Please train the model first.")
        
    df_single = pd.DataFrame([customer_dict])
    X_trans = preprocessor.transform(df_single)
    
    prob = float(model.predict_proba(X_trans)[0, 1])
    risk_level = get_risk_level(prob)
    prediction_label = "Likely to Churn" if prob >= 0.5 else "Likely to Stay"
    
    return {
        "prediction": prediction_label,
        "probability": round(prob * 100, 1),
        "raw_probability": prob,
        "risk_level": risk_level
    }


def explain_prediction(customer_dict):
    """
    Returns main business reasons why a customer may churn based on customer attributes.
    """
    reasons = []
    
    contract = customer_dict.get("Contract", "Month-to-month")
    monthly_charges = float(customer_dict.get("MonthlyCharges", 0))
    tenure = int(customer_dict.get("tenure", 0))
    tech_support = customer_dict.get("TechSupport", "No")
    internet_service = customer_dict.get("InternetService", "Fiber optic")
    payment_method = customer_dict.get("PaymentMethod", "Electronic check")
    
    if contract == "Month-to-month":
        reasons.append("Month-to-month contract (no long-term commitment)")
        
    if monthly_charges > 70.0:
        reasons.append(f"High monthly charges (₹{monthly_charges:.2f}/mo)")
        
    if tenure <= 12:
        reasons.append(f"Short tenure ({tenure} months with service)")
        
    if tech_support in ["No", "No internet service"]:
        reasons.append("No technical support subscription")
        
    if internet_service == "Fiber optic":
        reasons.append("Fiber optic internet plan (higher churn rate segment)")
        
    if payment_method == "Electronic check":
        reasons.append("Manual payment method (Electronic check)")
        
    if not reasons:
        reasons.append("General customer account profile characteristics")
        
    return reasons[:4]


def predict_batch(df):
    """
    Performs churn prediction across an entire customer DataFrame.
    """
    model, preprocessor, _ = load_model()
    if model is None or preprocessor is None:
        raise ValueError("Model or preprocessor not found. Please train the model first.")
        
    df_clean = df.copy()
    X = df_clean.drop(columns=["Churn", "customerID"], errors="ignore")
    
    X_trans = preprocessor.transform(X)
    probabilities = model.predict_proba(X_trans)[:, 1]
    
    df_result = df_clean.copy()
    df_result["Churn Probability"] = np.round(probabilities * 100, 1)
    df_result["Risk"] = [get_risk_level(p) for p in probabilities]
    df_result["Suggested Action"] = [get_retention_action(row.to_dict())[0] for _, row in df_clean.iterrows()]
    
    return df_result
