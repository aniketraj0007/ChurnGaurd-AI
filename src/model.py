import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def train_all_models(X_train, y_train):
    """
    Trains multiple ML classification models for churn prediction.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    }
    
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
        
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05, random_state=42, verbose=0)
        
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models


def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluates trained models on the test set and calculates key classification metrics and confusion matrix.
    """
    results = []
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "Confusion Matrix": cm
        })
        
    comparison_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return comparison_df, results


def save_best_model(model, preprocessor, metadata):
    """Saves the trained champion model, preprocessor, and performance metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)


def load_model():
    """Loads saved model, preprocessor, and metadata."""
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return None, None, None
        
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
            
    return model, preprocessor, metadata
