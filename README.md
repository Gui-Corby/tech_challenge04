# Stock Price Prediction API (LSTM)

This project implements an end-to-end **machine learning pipeline** for **time series forecasting of stock prices**, using an **LSTM neural network**, with deployment via **FastAPI** and **production-style monitoring**.

---

## Project Objective

The goal of this project is to:

- Collect and preprocess historical stock price data  
- Train an **LSTM-based model** to capture temporal patterns  
- Export the trained model for inference  
- Deploy the model through a **REST API**  
- Expose **monitoring metrics** such as latency and resource usage  

---

## Data Collection & Preprocessing

- Historical stock data is collected from **Yahoo Finance** using `yfinance`
- The dataset includes **OHLCV** information:
  - Open, High, Low, Close, Volume
- Feature engineering includes:
  - Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - Log-transformed volume
- Data is split **chronologically** into train and test sets to avoid data leakage
- Feature scaling is applied **only on the training set**, and reused for inference

---

## Model

- Architecture: **LSTM Regressor**
- Framework: **PyTorch**
- Objective: **One-step-ahead price forecasting**
- Hyperparameters explored:
  - Sequence length
  - Learning rate
- Evaluation metrics:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
- Experiment tracking is done using **MLflow**

---

## Model Artifacts

After training, the following artifacts are saved:

- Trained LSTM model (`.pth`)
- Feature scalers (`joblib`)
- Model configuration (sequence length, features)

These artifacts are loaded by the API during startup for inference.

---

## API Deployment

The model is deployed using **FastAPI**.

### Available Endpoints

#### `GET /`
Health check endpoint.

#### `POST /predict`
Generates a **one-step-ahead price prediction** based on historical data.

**Input**:
- A sequence of historical OHLCV candles
- The sequence length must be **greater than or equal to the model’s `SEQ_LENGTH`**

**Example request**:
```json
{
  "history": [
    {
      "open": 248.1,
      "high": 252.3,
      "low": 246.8,
      "close": 250.9,
      "volume": 118345000
    }
  ]
}
```

**Example response**:
```json
{
  "predicted_close": 279.5286265497034,
  "last_close": 292.6,
  "latency_ms": 11.162541020894423
}
```

Where:

- `predicted_close` is the model’s one-step-ahead forecast

- `last_close` is the most recent observed closing price

- `latency_ms` is the end-to-end inference latency measured at request time


## Monitoring & Observability

The API includes **basic production-style monitoring**, exposing **Prometheus-compatible metrics** to track performance and resource usage.


## How to Run the Project

### 1) Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### 2) Install dependencies
`pip install -r requirements.txt`

### 3) Run the API
From the project root directory:
`uvicorn src.main:app --reload`

### 4) Access the application
API root (health check):
`http://127.0.0.1:8000/`

Interactive API documentation (Swagger UI):
`http://127.0.0.1:8000/docs`

Monitoring metrics (Prometheus format):
`http://127.0.0.1:8000/metrics`

## Project Structure

```text
.
├── LICENSE
├── README.md                         # Project overview + instructions (run API, endpoints, metrics)
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Ignore generated artifacts (mlruns/, __pycache__/, venv, etc.)
├── models/
│   ├── best_model.pth                # Final trained LSTM weights used by the API
│   ├── scaler_price.joblib           # Scaler fitted on training price features (used in inference)
│   └── scaler_volume.joblib          # Scaler fitted on training volume feature (used in inference)
├── notebooks/
│   └── tech_challenge04.ipynb        # Initial experimentation/EDA (not used in production API)
└── src/
    ├── __init__.py                   # Marks src as a package (enables `python -m src.train`)
    ├── main.py                       # FastAPI app entrypoint; includes routers + monitoring middleware
    ├── config.py                     # Central configuration (paths, symbol, seq_length, feature columns)
    ├── data_pipeline.py              # Data ingestion + feature engineering (SMA, RSI, log volume) + scaling
    ├── lstm_model.py                 # PyTorch LSTMRegressor definition
    ├── train.py                      # Training loop + hyperparameter search + artifact saving
    ├── evaluate.py                   # Evaluation metrics (MSE/MAE/RMSE) for model selection
    ├── api/
    │   └── future_price.py           # `POST /predict` endpoint (input validation + inference + latency)
    └── monitoring/
        ├── __init__.py
        ├── metrics.py                # Prometheus metric definitions
        ├── middleware.py             # Request latency + CPU/memory instrumentation
        └── routes.py                 # `GET /metrics` endpoint
```

### Training the model (optional)

To run the training script, execute it as a module from the project root:
`python -m src.train`
or
`python3 -m src.train`

## Next steps

While this project demonstrates a complete and coherent pipeline for stock price forecasting, from data preprocessing and model training to API deployment and monitoring, there are several directions for future improvements.

First, although the model outperforms simple baselines, the absolute prediction error (MAE and RMSE in USD) remains relatively high. This is expected given the highly noisy, non-stationary nature of financial time series and the fact that the model relies exclusively on price-derived features. Future work could incorporate additional exogenous variables, such as market indices, macroeconomic indicators, or cross-asset information, to improve predictive performance.

Second, the current implementation focuses on one-step-ahead point forecasts. Extending the model to multi-step forecasting or probabilistic outputs (e.g., confidence intervals) could provide more informative predictions for real-world use cases.

Third, hyperparameter exploration in this project was limited to a small grid of learning rates and sequence lengths. More systematic optimization strategies, such as Bayesian optimization or rolling-window cross-validation, could be applied to further refine the model.

Finally, from a production perspective, future iterations could enhance model monitoring by tracking prediction drift and performance degradation over time, as well as implementing automated retraining strategies to adapt the model to changing market regimes.
