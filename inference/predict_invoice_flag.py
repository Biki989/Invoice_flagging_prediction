import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "amount_diff",
    "amount_ratio"
]

def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)

def load_scaler(scaler_path: str = SCALER_PATH):
    return joblib.load(scaler_path)

def predict_invoice_flag(input_data):
    model = load_model()
    scaler = load_scaler()
    input_df = pd.DataFrame(input_data)
    input_df["amount_diff"]  = input_df["invoice_dollars"] - input_df["total_item_dollars"]
    input_df["amount_ratio"] = input_df["invoice_dollars"] / (input_df["total_item_dollars"] + 1)
    input_df = input_df[FEATURES]
    scaled_input = scaler.transform(input_df)
    preds = model.predict(scaled_input)
    return [{"Predicted_Flag": int(f)} for f in preds]

if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [10, 20],
        "invoice_dollars": [5000, 888888],
        "Freight": [200, 10],
        "total_item_quantity": [10, 20],
        "total_item_dollars": [4950, 4]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)