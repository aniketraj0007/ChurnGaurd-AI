import pytest
import pandas as pd
from src.preprocessing import clean_data, load_data, prepare_features


def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_clean_data():
    sample_data = pd.DataFrame({
        "customerID": [" 123 ", " 456 "],
        "MonthlyCharges": [50.0, 70.0],
        "tenure": [10, 0],
        "TotalCharges": [" 500.0 ", "  "],
        "Churn": ["Yes", "No"]
    })
    
    df_clean = clean_data(sample_data)
    assert df_clean["TotalCharges"].iloc[1] == 0.0
    assert df_clean["Churn"].tolist() == [1, 0]


def test_prepare_features():
    df = load_data()
    data_dict = prepare_features(df)
    assert "X_train" in data_dict
    assert "y_train" in data_dict
    assert len(data_dict["X_train"]) > 0
