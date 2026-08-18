import sys
from pathlib import Path

app_dir = Path(__file__).resolve().parent.parent
root_dir = app_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.preprocessing import load_data, clean_data
from src.prediction import predict_batch

st.set_page_config(page_title="Dashboard | ChurnGuard AI", page_icon="📊", layout="wide")

st.title("ChurnGuard AI")
st.subheader("Customer Churn Prediction & Retention System")
st.markdown("---")

# Load real dataset
df_raw = load_data()
df = clean_data(df_raw)

# Compute metrics dynamically from actual dataset and model
total_customers = len(df)
churned_count = int(df["Churn"].sum())
churn_rate = (churned_count / total_customers) * 100

# Predict on actual dataset to count Customers at Risk dynamically
try:
    df_pred = predict_batch(df.head(1000))  # Predict on dataset sample for fast real calculation
    customers_at_risk = int((df_pred["Risk"].isin(["High Risk", "Medium Risk"])).sum())
except Exception:
    customers_at_risk = churned_count

# Display Top 3 Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", f"{total_customers:,}")
with col2:
    st.metric("Customers at Risk", f"{customers_at_risk:,}")
with col3:
    st.metric("Churn Rate", f"{churn_rate:.1f}%")

st.markdown("---")

# Simple Churn Distribution Chart from actual dataset
st.markdown("### Churn Distribution")
churn_counts = df["Churn"].map({1: "Churned", 0: "Retained"}).value_counts().reset_index()
churn_counts.columns = ["Status", "Count"]

fig = px.pie(
    churn_counts, 
    names="Status", 
    values="Count", 
    color="Status",
    color_discrete_map={"Churned": "#EF553B", "Retained": "#636EFA"},
    hole=0.4
)
fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
st.plotly_chart(fig, use_container_width=True)
