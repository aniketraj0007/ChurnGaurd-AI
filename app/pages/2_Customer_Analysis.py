import sys
from pathlib import Path

app_dir = Path(__file__).resolve().parent.parent
root_dir = app_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
from src.prediction import predict_churn, explain_prediction
from src.retention import get_retention_action

st.set_page_config(page_title="Customer Analysis | ChurnGuard AI", page_icon="🔮", layout="wide")

st.title("Customer Analysis")
st.subheader("Customer Churn Prediction Engine")

st.markdown("Enter customer details below to run churn prediction using the trained machine learning model.")

# Input Form
with st.form(key="customer_prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Demographics & Account")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)

    with col2:
        st.markdown("#### Services Subscribed")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("#### Billing & Contract")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges (₹)", min_value=18.0, max_value=120.0, value=75.0, step=1.0)
        total_charges = st.number_input("Total Charges (₹)", min_value=0.0, max_value=9000.0, value=float(tenure * monthly_charges))

    submit_button = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

if submit_button:
    customer_dict = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        res = predict_churn(customer_dict)
        reasons = explain_prediction(customer_dict)
        actions = get_retention_action(customer_dict)

        # Standardize Risk Level display (Low / Medium / High)
        risk_raw = res["risk_level"]
        if "High" in risk_raw:
            risk_label = "High"
        elif "Medium" in risk_raw:
            risk_label = "Medium"
        else:
            risk_label = "Low"

        prediction_text = "Likely to Churn" if res["raw_probability"] >= 0.5 else "Not Likely to Churn"

        st.markdown("---")
        st.markdown("### Prediction Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric("Prediction", prediction_text)

        with res_col2:
            st.metric("Churn Probability", f"{res['probability']}%")

        with res_col3:
            st.metric("Risk Level", risk_label)

        st.markdown("---")
        col_reason, col_action = st.columns(2)

        with col_reason:
            st.markdown("#### Top Contributing Factors")
            for r in reasons:
                st.write(f"- {r}")

        with col_action:
            st.markdown("#### Suggested Retention Action")
            st.info(f"👉 **{actions[0]}**")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
