import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("churnguard_model.pkl")


# -----------------------------
# Title
# -----------------------------

st.title("🛡️ ChurnGuard AI")

st.subheader("Customer Churn Prediction & Retention System")

st.write(
    "Enter customer information below to predict the likelihood "
    "of customer churn and get a retention recommendation."
)


# -----------------------------
# Customer Information
# -----------------------------

st.header("Customer Information")


col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )


with col3:

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# -----------------------------
# Charges
# -----------------------------

st.header("Billing Information")

col4, col5 = st.columns(2)


with col4:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )


with col5:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=monthly_charges * tenure
    )


# -----------------------------
# Prediction Button
# -----------------------------

st.divider()

predict_button = st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
)


# -----------------------------
# Prediction
# -----------------------------

if predict_button:

    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # Prediction

    prediction = model.predict(customer_data)[0]

    probability = model.predict_proba(customer_data)[0][1]

    probability_percentage = probability * 100


    # Risk Level

    if probability < 0.30:
        risk_level = "LOW"

    elif probability < 0.60:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"


    # -----------------------------
    # Results
    # -----------------------------

    st.header("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Churn Probability",
            f"{probability_percentage:.2f}%"
        )


    with result_col2:

        st.metric(
            "Risk Level",
            risk_level
        )


    with result_col3:

        if prediction == 1:
            prediction_text = "Likely to Churn"
        else:
            prediction_text = "Likely to Stay"

        st.metric(
            "Prediction",
            prediction_text
        )


    # -----------------------------
    # Retention Recommendation
    # -----------------------------

    st.subheader("💡 Retention Recommendation")


    if risk_level == "HIGH":

        st.error("High-risk customer detected.")

        recommendations = [
            "Contact the customer personally",
            "Offer a personalized discount",
            "Provide priority technical support"
        ]

    elif risk_level == "MEDIUM":

        st.warning("Medium-risk customer detected.")

        recommendations = [
            "Send a personalized retention offer",
            "Recommend a suitable long-term plan",
            "Check customer satisfaction"
        ]

    else:

        st.success("Low-risk customer.")

        recommendations = [
            "Continue regular customer engagement",
            "Offer loyalty benefits",
            "Monitor customer activity"
        ]


    for recommendation in recommendations:

        st.write("✓", recommendation)