import sys
from pathlib import Path

app_dir = Path(__file__).resolve().parent.parent
root_dir = app_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.model import load_model

st.set_page_config(page_title="Model Performance | ChurnGuard AI", page_icon="📊", layout="wide")

st.title("Model Performance")
st.subheader("Trained Model Test Evaluation")

model, preprocessor, metadata = load_model()

if metadata and "metrics" in metadata:
    metrics = metadata["metrics"]
    model_name = metadata.get("best_model_name", "CatBoost")

    st.markdown(f"Displaying actual test set metrics for champion model: **{model_name}**")
    st.markdown("---")

    # Metrics Row: Accuracy, Precision, Recall, F1 Score
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics.get('Accuracy', 0) * 100:.2f}%")
    with col2:
        st.metric("Precision", f"{metrics.get('Precision', 0) * 100:.2f}%")
    with col3:
        st.metric("Recall", f"{metrics.get('Recall', 0) * 100:.2f}%")
    with col4:
        st.metric("F1 Score", f"{metrics.get('F1 Score', 0) * 100:.2f}%")

    st.markdown("---")
    st.markdown("### Confusion Matrix")

    # Retrieve real Confusion Matrix from metadata
    cm = metrics.get("Confusion Matrix", None)
    if cm is None and "comparison" in metadata:
        # Find champion model confusion matrix in comparison records
        for rec in metadata["comparison"]:
            if rec.get("Model") == model_name:
                cm = rec.get("Confusion Matrix")
                break

    if cm is not None:
        cm_array = np.array(cm)
        labels = ["Retained (0)", "Churned (1)"]
        
        # Plot Heatmap of Confusion Matrix
        fig = px.imshow(
            cm_array,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Label", y="Actual Label", color="Count")
        )
        fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(fig, use_container_width=True)

        col_tn, col_fp, col_fn, col_tp = st.columns(4)
        with col_tn:
            st.caption(f"True Negatives: **{cm_array[0][0]:,}**")
        with col_fp:
            st.caption(f"False Positives: **{cm_array[0][1]:,}**")
        with col_fn:
            st.caption(f"False Negatives: **{cm_array[1][0]:,}**")
        with col_tp:
            st.caption(f"True Positives: **{cm_array[1][1]:,}**")
    else:
        st.info("Confusion matrix details available after model re-training.")

else:
    st.warning("Model metadata not found. Please run model training using `python run.py --train`.")
