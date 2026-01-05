import time
import joblib
import pandas as pd
import torch

from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.lstm_model import LSTMRegressor
from src.data_pipeline import create_features, drop_null, transform_scalers
from src.config import (
    COLS_PRICE,
    COL_VOLUME,
    FEATURE_COLS,
    SEQ_LENGTH,
    MODEL_PATH,
    SCALER_PRICE_PATH,
    SCALER_VOLUME_PATH,
    MIN_CANDLES,
    FEATURE_WARMUP,
    RECOMMENDED_CANDLES
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
scaler_price = None
scaler_volume = None


@asynccontextmanager
async def lifespan(app):
    global model, scaler_price, scaler_volume

    scaler_price = joblib.load(SCALER_PRICE_PATH)
    scaler_volume = joblib.load(SCALER_VOLUME_PATH)

    model = LSTMRegressor(
        input_size=len(FEATURE_COLS),
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    yield

router = APIRouter(lifespan=lifespan)


class Candle(BaseModel):
    close: float
    high: float
    low: float
    open: float
    volume: float


class PredictRequest(BaseModel):
    history: list[Candle]


@router.post("/predict")
def predict_stock(req: PredictRequest):
    start_time = time.perf_counter()
    if len(req.history) < SEQ_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"history must have at least {SEQ_LENGTH} candles"
        )

    # JSON -> DataFrame
    df = pd.DataFrame([c.model_dump() for c in req.history])
    df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                       "close": "Close", "volume": "Volume"}, inplace=True)
    # Features
    df = create_features(df)
    df = drop_null(df, FEATURE_COLS)

    before = len(req.history)
    after = len(df)

    if len(df) < SEQ_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                "Insufficient historical data to generate a prediction. "
                f"Received {before} candles, but only {after} remain after feature engineering. "
                "Some technical indicators (e.g., SMA and RSI) require an initial warm-up period "
                "and produce NaN values that are automatically discarded. "
                f"The model requires at least {MIN_CANDLES} usable candles "
                f"(sequence length = {SEQ_LENGTH}, indicator warm-up = {FEATURE_WARMUP}). "
                f"Please provide more historical data (recommended: {RECOMMENDED_CANDLES}+ candles).")
        )

    # Scaling
    df_scaled, _ = transform_scalers(df, df, scaler_price, scaler_volume,
                                     COLS_PRICE, COL_VOLUME)

    # Mount the last window
    X = df_scaled[FEATURE_COLS].values
    X_last = X[-SEQ_LENGTH:]
    X_tensor = torch.tensor(X_last, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq, n_feat)

    # Prediction
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).item()

    # back to Dollars
    close_idx = COLS_PRICE.index("Close")
    close_usd = y_pred_scaled * scaler_price.scale_[close_idx] + scaler_price.mean_[close_idx]
    last_close = float(df["Close"].iloc[-1])

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "predicted_close": float(close_usd),
        "last_close": last_close,
        "latency_ms": latency_ms}
