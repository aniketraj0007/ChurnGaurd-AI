"""
ChurnGuard AI - Launcher Script
Command line interface to train models or launch the Streamlit web dashboard.
"""

import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import load_data, prepare_features
from src.model import train_all_models, evaluate_models, save_best_model


def run_training():
    """Executes model training & evaluation pipeline."""
    print("Loading customer dataset...")
    df = load_data()
    
    print("Preparing train-test features...")
    data_dict = prepare_features(df)
    
    print("Training candidate models (Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost)...")
    trained_models = train_all_models(data_dict["X_train"], data_dict["y_train"])
    
    print("Evaluating model metrics...")
    comparison_df, full_results = evaluate_models(trained_models, data_dict["X_test"], data_dict["y_test"])
    
    display_df = comparison_df.drop(columns=["Confusion Matrix"], errors="ignore")
    print("\n--- Model Evaluation Results ---")
    print(display_df.to_string(index=False))
    
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_metrics = comparison_df.iloc[0].to_dict()
    
    print(f"\nChampion Model Selected: {best_model_name} (ROC-AUC: {best_metrics['ROC-AUC']})")
    
    metadata = {
        "best_model_name": best_model_name,
        "metrics": best_metrics,
        "comparison": comparison_df.to_dict(orient="records"),
        "numerical_cols": data_dict["numerical_cols"],
        "categorical_cols": data_dict["categorical_cols"]
    }
    
    save_best_model(best_model, data_dict["preprocessor"], metadata)
    print("Model training complete. Saved artifacts to models/\n")


def run_app():
    """Launches interactive Streamlit dashboard."""
    import subprocess
    app_path = ROOT_DIR / "app" / "app.py"
    print(f"Launching Streamlit application from {app_path}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], cwd=ROOT_DIR)


def main():
    parser = argparse.ArgumentParser(description="ChurnGuard AI Launcher")
    parser.add_argument("--train", "-t", action="store_true", help="Train and evaluate ML models.")
    parser.add_argument("--app", "-a", action="store_true", help="Launch Streamlit dashboard.")
    
    args = parser.parse_args()
    
    if args.train:
        run_training()
    elif args.app:
        run_app()
    else:
        print("Running default pipeline: Training Model -> Launching App")
        run_training()
        run_app()


if __name__ == "__main__":
    main()
