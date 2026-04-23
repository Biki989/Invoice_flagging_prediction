# 🧾 Invoice Flagging & Freight Cost Prediction

> A machine-learning-powered web application that predicts freight charges and detects suspicious invoices in real time — built for procurement and logistics teams.

🔗 **Live Demo:** [Vercel Link](https://invoice-flagging.vercel.app/)

---

**Freight Cost Prediction** uses Linear Regression, Decision Tree, and Random Forest Regressors — the best performer by MAE is selected for inference.

**Invoice Anomaly Flagging** uses a Random Forest Classifier trained on engineered ratio features (`amount_diff`, `amount_ratio`), with hyperparameter tuning via grid search.

A lightweight rule layer catches logically impossible entries (negative values, extreme ratios) before the ML model runs, ensuring defence-in-depth beyond pure statistical prediction.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python · Flask |
| ML / Data | scikit-learn · pandas · NumPy |
| Serialisation | joblib (`.pkl` model files) |
| Frontend | HTML · CSS · Vanilla JavaScript |
| Deployment | Vercel (Python serverless runtime) |

---

## Project Structure

```
.
├── app.py                          # Flask application & API routes
├── requirements.txt                # Python dependencies
├── vercel.json                     # Vercel deployment config
│
├── models/
│   ├── predict_freight_model.pkl   # Saved regression model
│   ├── predict_flag_invoice.pkl    # Saved classification model
│   └── scaler.pkl                  # Feature scaler for invoice model
│
├── Freight_cost_prediction/        # Freight model training pipeline
│   ├── data_preprocessing.py
│   ├── train.py
│   └── model_evaluation.py
│
├── invoice_flagging/               # Invoice flagging training pipeline
│   ├── data_preprocessing.py
│   ├── train.py
│   └── modeling_evaluation.py
│
├── inference/                      # Standalone inference scripts
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── Notebooks/                      # Exploratory & training notebooks
│   ├── Predicting Freight Cost.ipynb
│   └── Invoice_flagging.ipynb
│
├── Data/
│   └── inventory.db                # SQLite dataset
│
├── templates/
│   └── index.html                  # Dashboard UI
│
└── static/
    ├── style.css
    └── script.js
```

---

## Getting Started

### Prerequisites

- Python **3.10** or higher

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Biki989/Invoice_flagging_prediction.git
cd Invoice_flagging_prediction

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
python app.py
```

The app starts at **http://127.0.0.1:5000**.  
Enable hot-reload during development:

```bash
FLASK_DEBUG=1 python app.py
```

---

## API Reference

### `POST /predict/freight`

Predict freight cost from an invoice dollar amount.

**Request**
```json
{
  "Dollars": 18500
}
```

**Response**
```json
{
  "predictions": [
    { "Predicted_Freight": 925 }
  ]
}
```

---

### `POST /predict/invoice`

Flag an invoice as **normal** (`0`) or **suspicious** (`1`).

**Request**
```json
{
  "invoice_quantity": 10,
  "invoice_dollars": 5000,
  "Freight": 200,
  "total_item_quantity": 10,
  "total_item_dollars": 4950
}
```

**Response**
```json
{
  "predictions": [
    { "Predicted_Flag": 0 }
  ]
}
```

**Request fields**

| Field | Type | Description |
|---|---|---|
| `invoice_quantity` | float | Number of units on the invoice |
| `invoice_dollars` | float | Total dollar amount on the invoice |
| `Freight` | float | Freight/shipping charge on the invoice |
| `total_item_quantity` | float | Sum of item quantities received |
| `total_item_dollars` | float | Sum of item dollar values received |

> `amount_diff` and `amount_ratio` are derived server-side — **do not include them in the request**.

---

## Deployment

The repository ships with a pre-configured `vercel.json` for one-command serverless deployment.

```bash
# Install the Vercel CLI (one time)
npm i -g vercel

# Deploy to production
vercel --prod
```

---

## Retraining Models

Each training pipeline follows the same three-step structure:

1. **`data_preprocessing.py`** — Loads and cleans raw data from `Data/inventory.db`
2. **`train.py`** — Trains the model and serialises the `.pkl` artefact to `models/`
3. **`model_evaluation.py`** — Prints performance metrics (MAE, accuracy, etc.)

For an interactive walkthrough of the full pipeline, open the corresponding notebook in `Notebooks/`.

---

## License

This project is provided for **educational and demonstration purposes** only.
