import torch
import numpy as np
from pathlib import Path
from data_processor import KANDataProcessor1
from kan_model import *
import matplotlib.pyplot as plt
import warnings
from visualizer import LiveVisualizer

warnings.filterwarnings("ignore", category=FutureWarning)


def plot_training_history(history):
    train_losses, val_losses = zip(*history)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_history.png")
    plt.close()


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output.squeeze(), y)
            total_loss += loss.item()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(output.squeeze().cpu().numpy())

    test_loss = total_loss / len(test_loader)
    return test_loss, np.array(y_true), np.array(y_pred)


def main():
    processor = KANDataProcessor1()
    visualizer = LiveVisualizer()
    X_data, y_target, config = processor.prepare_training_data(
        Path("Data/expression_data1.h5ad"), Path("Data/JUN_interactions.tsv"), "JUN"
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = prepare_data(X_data, y_target)

    # Train model
    history = train_with_early_stopping(
        model, train_loader, val_loader, optimizer, criterion, epochs
    )
    plot_training_history(history)
    visualizer.plot_training_metrics(history)

    # Evaluate on test set
    test_loss, y_true, y_pred = evaluate_model(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f}")
    model.auto_symbolic()
    print(model.symbolic_formula()[0])
    model.plot()
    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "test_loss": test_loss,
        },
        "final_model.pt",
    )


if __name__ == "__main__":
    main()
