SYMBOL = "TSLA"  # Tesla, Inc
START_DATE = "2023-01-1"
END_DATE = "2024-12-31"

COLS_PRICE = ["High", "Low", "Open", "SMA", "Close"]
COL_VOLUME = ["Volume_log"]

FEATURE_COLS = ["High", "Low", "Open", "Volume_log", "SMA", "RSI_scaled"]
TARGET_COL = "Close"

SEQ_LENGTH = 10
SMA_WINDOW = 10
RSI_WINDOW = 14

# Discarded initial lines from NaN
FEATURE_WARMUP = max(SMA_WINDOW - 1, RSI_WINDOW)  # 14
MIN_CANDLES = SEQ_LENGTH + FEATURE_WARMUP

RECOMMENDED_CANDLES = MIN_CANDLES + 10

MODEL_PATH = "models/model_lr0.001_seq10.pth"
SCALER_PRICE_PATH = "models/scaler_price.joblib"
SCALER_VOLUME_PATH = "models/scaler_volume.joblib"
