import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = SCRIPT_DIR.parent

from data_preprocessing import load_invoice_data, split_data, scale_features, apply_labels, add_features
from modeling_evaluation import train_random_forest, evaluate_classifier
import joblib

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "amount_diff",
    "amount_ratio"
]
TARGET = "flag_invoice"

def main():
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    df = load_invoice_data()
    df = apply_labels(df)
    df = add_features(df)
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, str(models_dir / 'scaler.pkl')
    )
    grid_search = train_random_forest(X_train_scaled, y_train)
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )
    joblib.dump(
        grid_search.best_estimator_,
        str(models_dir / 'predict_flag_invoice.pkl')
    )

if __name__ == "__main__":
    main()