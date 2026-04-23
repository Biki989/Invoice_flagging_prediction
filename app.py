import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

freight_model = invoice_model = scaler = None

def load_models():
    global freight_model, invoice_model, scaler
    try:
        freight_model = joblib.load("models/predict_freight_model.pkl")
        invoice_model = joblib.load("models/predict_flag_invoice.pkl")
        scaler        = joblib.load("models/scaler.pkl")
    except FileNotFoundError as e:
        print(f"[ERROR] Model file not found: {e}")
        raise

load_models()

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "amount_diff",
    "amount_ratio"
]

FREIGHT_FEATURES = ["Dollars"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict/freight", methods=["POST"])
def predict_freight():
    try:
        data = request.get_json()
        if "Dollars" not in data:
            return jsonify({"error": "Missing field: Dollars"}), 400

        input_df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        input_df = input_df[FREIGHT_FEATURES]
        input_df["Predicted_Freight"] = freight_model.predict(input_df).round()

        return jsonify({"predictions": input_df[["Predicted_Freight"]].to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def absurd_value_check(row):
    inv_dollars  = float(row["invoice_dollars"])
    item_dollars = float(row["total_item_dollars"])
    inv_qty      = float(row["invoice_quantity"])
    item_qty     = float(row["total_item_quantity"])
    freight      = float(row["Freight"])

    if inv_dollars < 0 or item_dollars < 0 or freight < 0:
        return 1

    if inv_qty <= 0 or item_qty <= 0:
        return 1

    if item_dollars > 0 and inv_dollars / item_dollars > 10:
        return 1

    if inv_dollars > 0 and item_dollars / inv_dollars > 10:
        return 1

    if inv_dollars > 0 and freight > inv_dollars:
        return 1
    if item_dollars > 0 and freight > item_dollars:
        return 1

    if inv_qty > 0 and item_qty / inv_qty > 10:
        return 1
    if item_qty > 0 and inv_qty / item_qty > 10:
        return 1

    return 0


@app.route("/predict/invoice", methods=["POST"])
def predict_invoice():
    try:
        data = request.get_json()

        required = ["invoice_quantity", "invoice_dollars", "Freight",
                    "total_item_quantity", "total_item_dollars"]
        for f in required:
            if f not in data:
                return jsonify({"error": f"Missing field: {f}"}), 400

        input_df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)

        absurd_flags = input_df.apply(absurd_value_check, axis=1).values

        input_df["amount_diff"]  = input_df["invoice_dollars"] - input_df["total_item_dollars"]
        input_df["amount_ratio"] = input_df["invoice_dollars"] / (input_df["total_item_dollars"] + 1)
        input_df = input_df[FEATURES]

        scaled = scaler.transform(input_df)

        preds = invoice_model.predict(scaled)

        final = np.maximum(preds, absurd_flags)

        return jsonify({"predictions": [{"Predicted_Flag": int(f)} for f in final]})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)