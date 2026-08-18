import sys
from pathlib import Path

# Add project root directory to path
app_dir = Path(__file__).resolve().parent
root_dir = app_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
from src.model import load_model
from src.preprocessing import load_data

st.set_page_config(
    page_title="ChurnGuard AI | Customer Churn System",
    page_icon="📊",
    layout="wide"
)

# Sidebar
st.sidebar.title("ChurnGuard AI")
st.sidebar.caption("Customer Churn Prediction & Retention System")
st.sidebar.markdown("---")

st.sidebar.markdown("""
### Modules
Use the sidebar pages to navigate:
- 📊 **Dashboard**
- 🔮 **Customer Analysis**
- 📈 **Model Performance**
""")

st.sidebar.markdown("---")

# Main Header
st.title("ChurnGuard AI")
st.subheader("Customer Churn Prediction & Retention System")

st.markdown("""
Welcome to **ChurnGuard AI**. This is a machine learning application that predicts whether a customer is likely to leave a service based on tenure, contract type, monthly charges, and subscribed services.
""")

st.markdown("---")

# System Status Section
model, preprocessor, metadata = load_model()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📁 Dataset Status")
    try:
        df = load_data()
        st.success(f"Dataset Loaded: **{len(df):,} accounts**")
    except Exception as e:
        st.error(f"Dataset Error: {e}")

with col2:
    st.markdown("### ⚙️ Model Status")
    if model is not None and metadata:
        model_name = metadata.get("best_model_name", "Trained Model")
        acc = metadata.get("metrics", {}).get("Accuracy", "N/A")
        st.success(f"Trained Model Ready: **{model_name}**")
    else:
        st.warning("Model Not Trained. Please train the model using `python run.py --train`.")

st.markdown("---")
st.info("👈 **Please select a module from the sidebar navigation menu to begin.**")
