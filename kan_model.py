from kan import *
import torch
import numpy as np
from pathlib import Path
from data_processor import KANDataProcessor1
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
model = KAN(
    width=config["width"],
    grid=config["grid"],
    k=config["k"],
    seed=config["seed"],
    device=device,
)

# Training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
criterion = torch.nn.MSELoss()
epochs = 5


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
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
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            loss = train_batch(model, X_batch, y_batch, optimizer, criterion)
            train_loss += loss

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                val_loss += criterion(
                    model(X.to(device)).squeeze(), y.to(device)
                ).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history.append((train_loss, val_loss))

        pbar.set_postfix(
            {"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"}
        )

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

    return history


def train_batch(model, X, y, optimizer, criterion):
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output.squeeze(), y)
    loss.backward()
    optimizer.step()
    return loss.item()
