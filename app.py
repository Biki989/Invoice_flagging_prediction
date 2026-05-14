import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

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
]

FREIGHT_FEATURES = ["Dollars"]


def sanity_check(data: dict) -> tuple:
    """Returns (flag, reason). flag=1 means auto-flag."""
    inv_qty = data["invoice_quantity"]
    inv_dol = data["invoice_dollars"]
    freight = data["Freight"]
    tot_qty = data["total_item_quantity"]
    tot_dol = data["total_item_dollars"]

    # 1. Negative values
    if any(v < 0 for v in [inv_qty, inv_dol, freight, tot_qty, tot_dol]):
        return 1, "Negative values detected"

    # 2. Zero dollars with non-zero quantity
    if inv_dol == 0 and inv_qty > 0:
        return 1, "Zero invoice dollars with non-zero quantity"

    # 3. Freight exceeds invoice
    if freight > inv_dol and inv_dol > 0:
        return 1, "Freight exceeds invoice value"

    # 4. Extreme dollar ratio (invoice vs PO)
    ratio = inv_dol / (tot_dol + 1)
    if ratio > 10:
        return 1, f"Invoice/PO dollar ratio is {ratio:.1f}x"

    # 5. Extreme quantity ratio
    qty_ratio = inv_qty / (tot_qty + 1)
    if qty_ratio > 10:
        return 1, f"Invoice/PO quantity ratio is {qty_ratio:.1f}x"

    return 0, ""


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




@app.route("/predict/invoice", methods=["POST"])
def predict_invoice():
    try:
        data = request.get_json()

        required = ["invoice_quantity", "invoice_dollars", "Freight",
                    "total_item_quantity", "total_item_dollars"]
        for f in required:
            if f not in data:
                return jsonify({"error": f"Missing field: {f}"}), 400

        # ── Rule-based pre-filter ──
        rule_flag, rule_reason = sanity_check(data)

        # ── ML prediction ──
        input_df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        input_df = input_df[FEATURES]

        scaled = scaler.transform(input_df)
        ml_flag = int(invoice_model.predict(scaled)[0])

        # ── Final decision: either layer can flag ──
        final_flag = max(ml_flag, rule_flag)
        reason = rule_reason if rule_flag else ""

        return jsonify({"predictions": [{
            "Predicted_Flag": final_flag,
            "rule_flag": rule_flag,
            "ml_flag": ml_flag,
            "reason": reason
        }]})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)