from kan import *
import torch
import numpy as np
from pathlib import Path
from data_processor import KANDataProcessor1
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_FILE = "training_log.txt"


def log_training_details(params, max_memory, final_val_loss, history, config):
    with open(LOG_FILE, "a") as f:
        f.write("=" * 50 + "\n")
        f.write(
            f"Configuration: width={config['width']}, grid={config['grid']}, k={config['k']}\n"
        )
        f.write(f"Total Parameters: {params}\n")
        f.write(f"Maximum Memory Consumed: {max_memory} GB\n")
        f.write(f"Final Validation Loss: {final_val_loss}\n")
        f.write("Epoch-wise Training Statistics:\n")
        for epoch, (
            train_loss,
            val_loss,
            test_loss,
            val_r2,
            val_rmse,
            val_mae,
        ) in enumerate(history, 1):
            f.write(
                f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, Test Loss ={test_loss:.6f},"
                f"Val R2 = {val_r2:.6f}, Val RMSE = {val_rmse:.6f}, Val MAE = {val_mae:.6f}\n"
            )
        f.write("=" * 50 + "\n")


class KANDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_data(X, y, batch_size=32):
    # Split indices
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    # Create datasets
    train_loader = DataLoader(
        KANDataset(X[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(KANDataset(X[val_idx], y[val_idx]), batch_size=batch_size)
    test_loader = DataLoader(
        KANDataset(X[test_idx], y[test_idx]), batch_size=batch_size
    )

    return train_loader, val_loader, test_loader


a = KANDataProcessor1()
y = a.prepare_training_data(
    Path("Data/expression_data1.h5ad"), Path("Data/JUN_interactions.tsv"), "JUN"
)
print(y)

# Unpack the data
X_data, y_target, config = y


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    epochs,
    patience=5,
    min_delta=1e-4,
):
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    pbar = tqdm(range(epochs), desc="Training", ncols=100, position=0, leave=True)
    for epochs in pbar:
        # Training phase
        model.train()
        train_loss, train_r2, train_rmse, train_mae = 0, 0, 0, 0
        for X_batch, y_batch in train_loader:
            loss = train_batch(model, X_batch, y_batch, optimizer, criterion)
            train_loss += loss

        # Validation phase
        model.eval()
        val_loss, val_r2, val_rmse, val_mae = 0, 0, 0, 0
        test_loss, test_r2, test_rmse, test_mae = 0, 0, 0, 0

        with torch.no_grad():
            for X, y in val_loader:
                predictions = model(X.to(device)).squeeze()
                y = y.to(device)
                val_loss += criterion(predictions, y).item()
                val_r2 += r2_score(y, predictions)
                val_rmse += rmse(y, predictions)
                val_mae += mae(y, predictions)

            for X, y in test_loader:
                predictions = model(X.to(device)).squeeze()
                y = y.to(device)
                test_loss += criterion(predictions, y).item()
                test_r2 += r2_score(y, predictions)
                test_rmse += rmse(y, predictions)
                test_mae += mae(y, predictions)

        # Normalize metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        test_loss /= len(test_loader)
        val_r2 /= len(val_loader)
        val_rmse /= len(val_loader)
        val_mae /= len(val_loader)
        test_r2 /= len(test_loader)
        test_rmse /= len(test_loader)
        test_mae /= len(test_loader)

        # Store metrics
        history.append((train_loss, val_loss, test_loss, val_r2, val_rmse, val_mae))

        # Update progress bar
        pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_r2": f"{val_r2:.4f}",
                "val_rmse": f"{val_rmse:.4f}",
            }
        )

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("\nEarly stopping triggered")
            model.load_state_dict(torch.load("best_model.pt"))
            break

    # Log final details
    total_params = sum(p.numel() for p in model.parameters())
    max_memory = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.05
    )
    log_training_details(total_params, max_memory, best_val_loss, history, config)
    return history


def train_batch(model, X, y, optimizer, criterion):
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output.squeeze(), y)
    loss.backward()
    optimizer.step()
    return loss.item()


def r2_score(y_true, y_pred):
    total_sum_squares = torch.sum((y_true - torch.mean(y_true)) ** 2)
    residual_sum_squares = torch.sum((y_true - y_pred) ** 2)
    return 1 - (residual_sum_squares / total_sum_squares)


def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))
