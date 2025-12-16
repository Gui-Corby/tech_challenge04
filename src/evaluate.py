import mlflow
import torch
import torch.nn as nn
from data_pipeline import (load_data,
                           create_features,
                           drop_null,
                           fit_scalers,
                           transform_scalers,
                           create_sequences,
                           to_tensors,
                           make_dataloaders)

from train import (COLS_PRICE,
                   COL_VOLUME,
                   FEATURE_COLS,
                   TARGET_COL,
                   split_train_test)

from lstm_model import LSTMRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model_path: str = "models/first_model.pth"):
    df = load_data()
    df = create_features(df)
    df = drop_null(df, FEATURE_COLS)

    train_df, test_df = split_train_test(df)

    scaler_price, scaler_volume = fit_scalers(train_df, COLS_PRICE, COL_VOLUME)
    train_df_scaled, test_df_scaled = transform_scalers(
        train_df, test_df,
        scaler_price,
        scaler_volume,
        COLS_PRICE,
        COL_VOLUME
    )

    X_train = train_df_scaled[FEATURE_COLS].values
    y_train = train_df_scaled[TARGET_COL].values
    X_test = test_df_scaled[FEATURE_COLS].values
    y_test = test_df_scaled[TARGET_COL].values

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length=20)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length=20)

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = to_tensors(
        X_train_seq, y_train_seq,
        X_test_seq, y_test_seq
    )

    train_loader, test_loader = make_dataloaders(
        X_train_tensor, y_train_tensor,
        X_test_tensor, y_test_tensor
    )

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = LSTMRegressor(len(FEATURE_COLS),
                          hidden_size=64,
                          num_layers=2,
                          dropout=0.2).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    model.eval()
    criterion = nn.MSELoss()

    test_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            test_loss += loss.item() * X_batch.size(0)

    test_loss /= len(test_loader.dataset)
    print(f"Test MSE: {test_loss:.6f}")
    mlflow.log_metric("test_MSE", test_loss)


if __name__ == "__main__":
    evaluate_model()
