import os
import pandas as pd
import mlflow
import mlflow.pytorch
import joblib
import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient
from data_pipeline import (load_data,
                           create_features,
                           drop_null,
                           fit_scalers,
                           transform_scalers,
                           create_sequences,
                           to_tensors,
                           make_dataloaders)

from config import (
    COLS_PRICE,
    COL_VOLUME,
    FEATURE_COLS,
    TARGET_COL,
    SEQ_LENGTH,
    MODEL_PATH,
    SCALER_PRICE_PATH,
    SCALER_VOLUME_PATH)

from lstm_model import LSTMRegressor


mlflow.set_experiment("lstm_stock_forecast")


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.8):

    df = df.copy()

    train_size = int(len(df) * train_ratio)

    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    return train_df, test_df


df = load_data()
df = create_features(df)
df = drop_null(df, FEATURE_COLS)

train_df, test_df = split_train_test(df)
scaler_price, scaler_volume = fit_scalers(train_df, COLS_PRICE, COL_VOLUME)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler_price, SCALER_PRICE_PATH)
joblib.dump(scaler_volume, SCALER_VOLUME_PATH)

train_df_scaled, test_df_scaled = transform_scalers(train_df, test_df, scaler_price, scaler_volume,
                                                    COLS_PRICE, COL_VOLUME)

X_train = train_df_scaled[FEATURE_COLS].values
y_train = train_df_scaled[TARGET_COL].values
X_test = test_df_scaled[FEATURE_COLS].values
y_test = test_df_scaled[TARGET_COL].values


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(device)


def call_model(lr=1e-3, num_epochs=50, seq_length=SEQ_LENGTH,  # Need to fix this seq_lenght!
               model_path=MODEL_PATH):
    with mlflow.start_run():
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("seq_length", SEQ_LENGTH)

        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = to_tensors(X_train_seq, y_train_seq, X_test_seq, y_test_seq)

        baseline_zero_mse = torch.mean(y_test_tensor**2).item()

        mlflow.log_metric("baseline_zero_test_mse", baseline_zero_mse)

        y_true = torch.tensor(y_test_seq[1:], dtype=torch.float32)
        y_naive = torch.tensor(y_test_seq[:-1], dtype=torch.float32)

        baseline_naive_mse = torch.mean((y_naive - y_true) ** 2).item()

        train_loader, test_loader = make_dataloaders(
            X_train_tensor, y_train_tensor,
            X_test_tensor, y_test_tensor)

        model = LSTMRegressor(len(FEATURE_COLS),
                              hidden_size=64,
                              num_layers=2,
                              dropout=0.2).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_loss = float("inf")

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(train_loader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs} - train MSE: {epoch_loss:.6f}")

            mlflow.log_metric("train_mse", epoch_loss, step=epoch + 1)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), model_path)

        print(f"\nBaseline ZERO test MSE: {baseline_zero_mse:.6f}")
        print(f"\nBaseline naive: {baseline_naive_mse:.6f}")

        model.eval()
        test_loss = 0.0

        close_idx = COLS_PRICE.index("Close")
        mu = float(scaler_price.mean_[close_idx])
        sigma = float(scaler_price.scale_[close_idx])

        sum_abs_usd = 0.0
        sum_sq_usd = 0.0
        n = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)

                loss = criterion(y_pred, y_batch)
                test_loss += loss.item() * X_batch.size(0)

                # shapes: (batch, 1) -> (batch,)
                y_pred_scaled = y_pred.squeeze(1)
                y_true_scaled = y_batch.squeeze(1)

                # back tou USD
                y_pred_usd = y_pred_scaled * sigma + mu
                y_true_usd = y_true_scaled * sigma + mu

                diff = y_pred_usd - y_true_usd
                sum_abs_usd += diff.abs().sum().item()
                sum_sq_usd += (diff ** 2).sum().item()
                n += y_true_usd.numel()

        test_loss /= len(test_loader.dataset)

        test_mae_usd = sum_abs_usd / n
        test_rmse_usd = (sum_sq_usd / n) ** 0.5

        print(f"\nTest MSE (scaled): {test_loss:.6f}")
        print(f"Test MAE (USD): {test_mae_usd:.4f}")
        print(f"Test RMSE (USD): {test_rmse_usd:.4f}")

        mlflow.pytorch.log_model(model, "model")
        mlflow.log_metric("test_mse", test_loss)
        mlflow.log_metric("test_mae_usd", test_mae_usd)
        mlflow.log_metric("test_rmse_usd", test_rmse_usd)

    return model


if __name__ == "__main__":
    lrs = [1e-2, 1e-3, 1e-4]
    seq_lengths = [10, 20, 30]

    for lr in lrs:
        for seq in seq_lengths:
            print(f"Running experiment with lr={lr} e seq_length={seq}")

            call_model(
                lr=lr,
                num_epochs=50,
                model_path=f"models/model_lr{lr}_seq{seq}.pth",
                seq_length=seq
            )

    client = MlflowClient()

    experiment = client.get_experiment_by_name("lstm_stock_forecast")

    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_rmse_usd ASC"],
        max_results=1
    )[0]

    best_run_id = best_run.info.run_id
    best_rmse = best_run.data.metrics["test_rmse_usd"]

    print(f"Best run id: {best_run_id}")
    print(f"Best test RMSE (USD): {best_rmse:.4f}")
