COLS_PRICE = ["High", "Low", "Open", "SMA", "Close"]
COL_VOLUME = ["Volume_log"]

FEATURE_COLS = ["High", "Low", "Open", "Volume_log", "SMA", "RSI_scaled"]
TARGET_COL = "Close"

SEQ_LENGTH = 20
MODEL_PATH = "models/first_model.pth"
SCALER_PRICE_PATH = "models/scaler_price.joblib"
SCALER_VOLUME_PATH = "models/scaler_volume.joblib"
