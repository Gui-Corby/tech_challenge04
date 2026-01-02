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

### 2) Install dependencies
pip install -r requirements.txt

### 3) Run the API
From the project root directory:
```bash
uvicorn src.main:app --reload

### 4) Access the application
API root (health check):
http://127.0.0.1:8000/

Interactive API documentation (Swagger UI):
http://127.0.0.1:8000/docs

Monitoring metrics (Prometheus format):
http://127.0.0.1:8000/metrics
