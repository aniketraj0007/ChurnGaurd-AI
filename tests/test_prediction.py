import pytest
from src.prediction import predict_churn, explain_prediction


def test_explain_prediction():
    cust = {
        "Contract": "Month-to-month",
        "MonthlyCharges": 80.0,
        "tenure": 3,
        "TechSupport": "No"
    }
    reasons = explain_prediction(cust)
    assert isinstance(reasons, list)
    assert len(reasons) > 0
    assert any("Month-to-month" in r for r in reasons)
