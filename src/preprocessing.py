import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "telco_customer_churn.csv"


def load_data(file_path=None):
    """Loads the customer churn dataset from CSV."""
    path = file_path if file_path else DATA_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file not found at {path}")
    return pd.read_csv(path)


def clean_data(df):
    """
    Cleans raw customer churn data:
    - Strips column names and string spaces
    - Converts TotalCharges to numeric and fills missing values
    - Converts target variable Churn (Yes/No) to 1/0 binary
    """
    df = df.copy()
    
    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]
    
    # Strip whitespace from string values
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Convert TotalCharges to float and handle empty strings
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Fill missing TotalCharges with tenure * MonthlyCharges
        missing_mask = df["TotalCharges"].isna()
        if missing_mask.sum() > 0:
            df.loc[missing_mask, "TotalCharges"] = df.loc[missing_mask, "MonthlyCharges"] * df.loc[missing_mask, "tenure"]
            
    # Convert target Churn to binary
    if "Churn" in df.columns:
        if df["Churn"].dtype == object or isinstance(df["Churn"].iloc[0], str):
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0).astype(int)
            
    return df


def prepare_features(df, test_size=0.2, random_state=42):
    """
    Prepares X and y, splits data into train/test sets,
    and applies scaling + one-hot encoding without data leakage.
    """
    df_clean = clean_data(df)
    
    # Target variable and feature matrix
    y = df_clean["Churn"]
    X = df_clean.drop(columns=["Churn", "customerID"], errors="ignore")
    
    # Feature types
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [col for col in X.columns if col not in numerical_cols]
    
    # Train-test split (stratified because churn class is imbalanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ]
    )
    
    # Fit on train, transform both
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # Get feature names after encoding
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    feature_names = numerical_cols + encoded_cat_names
    
    return {
        "X_train": X_train_trans,
        "X_test": X_test_trans,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "X_raw_train": X_train,
        "X_raw_test": X_test
    }
