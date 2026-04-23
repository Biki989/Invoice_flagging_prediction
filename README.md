# Invoice Flagging & Freight Cost Prediction

A machine-learning-powered Flask web application that **predicts freight costs** from invoice dollar amounts and **flags suspicious invoices** using anomaly detection. Built for procurement and logistics teams to perform real-time risk analysis on incoming invoices.

🔗 **Live Demo:** [https://your-app.vercel.app](https://invoice-flagging.vercel.app/)

---

## Features

- **Freight Cost Prediction** — Predicts expected freight charges based on invoice dollar value using a trained regression model.
- **Invoice Anomaly Flagging** — Classifies invoices as normal or suspicious by analysing quantity/dollar mismatches, freight anomalies, and derived ratio features through a trained ML classifier.
- **Absurd-Value Sanity Checks** — A lightweight rule layer catches genuinely impossible entries (negative values, extreme ratios) before the ML model runs.
- **Interactive Dashboard** — A single-page frontend for submitting invoices and viewing predictions in real time.
- **Deployable on Vercel** — Ships with a `vercel.json` for one-click serverless deployment.

---

## Tech Stack

| Layer       | Technology                          |
| ----------- | ----------------------------------- |
| Backend     | Python · Flask                      |
| ML / Data   | scikit-learn · pandas · NumPy       |
| Serialisation | joblib (`.pkl` model files)       |
| Frontend    | HTML · CSS · JavaScript             |
| Deployment  | Vercel (Python runtime)             |

---

## Project Structure

```
├── app.py                        # Flask application & API routes
├── requirements.txt              # Python dependencies
├── vercel.json                   # Vercel deployment config
│
├── models/                       # Serialised model artefacts
│   ├── predict_freight_model.pkl # Freight cost regression model
│   ├── predict_flag_invoice.pkl  # Invoice flag classifier
│   └── scaler.pkl                # Feature scaler for invoice model
│
├── Freight_cost_prediction/      # Freight model training pipeline
│   ├── data_preprocessing.py
│   ├── train.py
│   └── model_evaluation.py
│
├── invoice_flagging/             # Invoice flag training pipeline
│   ├── data_preprocessing.py
│   ├── train.py
│   └── modeling_evaluation.py
│
├── inference/                    # Standalone inference scripts
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── Notebooks/                    # Exploratory / training notebooks
│   ├── Predicting Freight Cost.ipynb
│   └── Invoice_flagging.ipynb
│
├── Data/                         # Dataset (SQLite DB)
│   └── inventory.db
│
├── templates/
│   └── index.html                # Dashboard UI
│
└── static/
    ├── style.css                 # Dashboard styles
    └── script.js                 # Dashboard logic
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**

### Installation

```bash
# Clone the repository
git clone https://github.com/Biki989/Invoice_flagging_prediction.git
cd Invoice_flagging_prediction

# Create & activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
python app.py
```

The app starts on **http://127.0.0.1:5000** by default.  
Set `FLASK_DEBUG=1` to enable hot-reload during development.

---

## API Reference

### `POST /predict/freight`

Predict freight cost from an invoice dollar amount.

**Request Body**

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

Flag an invoice as normal (`0`) or suspicious (`1`).

**Request Body**

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

| Field                | Type  | Description                            |
| -------------------- | ----- | -------------------------------------- |
| `invoice_quantity`   | float | Number of units on the invoice         |
| `invoice_dollars`    | float | Total dollar amount on the invoice     |
| `Freight`            | float | Freight/shipping charge on the invoice |
| `total_item_quantity`| float | Sum of item quantities received        |
| `total_item_dollars` | float | Sum of item dollar values received     |

> `amount_diff` and `amount_ratio` are computed server-side — you do **not** need to send them.

---

## Deployment

### Vercel

The project includes a pre-configured `vercel.json`:

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

---

## Retraining Models

Training pipelines live in `Freight_cost_prediction/` and `invoice_flagging/`. Each contains:

1. **`data_preprocessing.py`** — Loads and cleans raw data from `Data/inventory.db`.
2. **`train.py`** — Trains the model and saves the `.pkl` artefact to `models/`.
3. **`model_evaluation.py`** — Evaluates model performance metrics.

Jupyter notebooks in `Notebooks/` provide interactive walkthroughs of the full training process.

---

## License

This project is provided for educational and demonstration purposes.
