import torch


def evaluate(model, test_loader, criterion, device, scaler_price, cols_price):
    model.eval()
    test_loss = 0.0

    close_idx = cols_price.index("Close")
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

            y_pred_scaled = y_pred.squeeze(0)
            y_true_scaled = y_batch.squeeze(0)

            y_pred_usd = y_pred_scaled * sigma + mu
            y_true_usd = y_true_scaled * sigma + mu

            diff = y_pred_usd - y_true_usd
            sum_abs_usd += diff.abs().sum().item()
            sum_sq_usd += (diff ** 2).sum().item()
            n += y_true_usd.numel()
    
    return {
        "test_mse_scaled": test_loss / len(test_loader.dataset),
        "test_mae_usd": sum_abs_usd / n,
        "test_rmse_usd": (sum_sq_usd / n) ** 0.5
    }
    
