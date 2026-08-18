import pytest
from src.retention import get_risk_level, get_retention_action


def test_get_risk_level():
    assert get_risk_level(0.75) == "High Risk"
    assert get_risk_level(0.45) == "Medium Risk"
    assert get_risk_level(0.15) == "Low Risk"


def test_get_retention_action():
    cust_high_risk = {
        "Contract": "Month-to-month",
        "MonthlyCharges": 85.0,
        "tenure": 5,
        "TechSupport": "No"
    }
    actions = get_retention_action(cust_high_risk)
    assert isinstance(actions, list)
    assert len(actions) > 0
    assert any("contract upgrade" in a.lower() for a in actions)
