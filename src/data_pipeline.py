import pandas as pd
# import pandas_ta as ta
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.config import (
    SYMBOL,
    START_DATE,
    END_DATE
)


# Load data
def load_data() -> pd.DataFrame:
    symbol = SYMBOL  # Tesla, Inc
    start_date = START_DATE
    end_date = END_DATE

    df = yf.download(symbol, start=start_date, end=end_date)
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.rename(columns={'Close_TSLA': 'Close', 'High_TSLA': 'High',
                       'Low_TSLA': 'Low', 'Open_TSLA': 'Open',
                       'Volume_TSLA': 'Volume'}, inplace=True)

    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# Create features
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features like RSI, SMA, Volume_log"""

    df = df.copy()

    # 10-period Simple Moving Average (SMA)
    df['SMA'] = df["Close"].rolling(window=10).mean()

    # 14-period Relative Strength Index (RSI)
    df['RSI'] = compute_rsi(df['Close'], window=14)

    # Applying logarithm to volume values
    df['Volume_log'] = np.log1p(df['Volume'])

    df['RSI_scaled'] = df['RSI'] / 100.0

    return df


def drop_null(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df.copy()

    required_cols = set(feature_cols + ["Close"])
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"drop_null: missing columns: {missing}")

    df = df.dropna(subset=required_cols)

    return df


def fit_scalers(train_df: pd.DataFrame, cols_price: list, cols_volume: list):
    scaler_price = StandardScaler()
    scaler_price.fit(train_df[cols_price])

    scaler_volume = MinMaxScaler()
    scaler_volume.fit(train_df[cols_volume])

    return scaler_price, scaler_volume


def transform_scalers(train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      scaler_price,
                      scaler_volume,
                      cols_price: list,
                      cols_volume: list):

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[cols_price] = scaler_price.transform(train_df[cols_price])
    test_df[cols_price] = scaler_price.transform(test_df[cols_price])

    train_df[cols_volume] = scaler_volume.transform(train_df[cols_volume])
    test_df[cols_volume] = scaler_volume.transform(test_df[cols_volume])

    return train_df, test_df


# Generate seq_length size windows for LSTM
def create_sequences(X, y, seq_length: int):
    X_seq, y_seq = [], []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])

    return np.array(X_seq), np.array(y_seq)


# Transforming values to tensors
def to_tensors(X_train, y_train, X_test, y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


# Dataloaders
def make_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True)

    return train_loader, test_loader
