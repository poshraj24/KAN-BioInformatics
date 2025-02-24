import torch
import numpy as np
from pathlib import Path
from data_processor import KANDataProcessor1
from kan_model import *
import matplotlib.pyplot as plt
import warnings
from visualizer import LiveVisualizer


warnings.filterwarnings("ignore", message="meta NOT subset.*")
warnings.filterwarnings("ignore", category=FutureWarning)


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


def log_symbolic_formula(formula_tuple):
    """Log symbolic formula to training log file."""
    with open(LOG_FILE, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("Symbolic Formula:\n")
        f.write("-" * 20 + "\n")
        # Convert tuple elements to string and join them
        formula_str = "\n".join(str(item) for item in formula_tuple)
        f.write(formula_str + "\n")
        f.write("=" * 50 + "\n\n")


def save_plot(model, filename="model_plot.png", dpi=1200):
    """Save the KAN architecture plot to a file."""
    model.plot()  # This creates the plot
    plt.savefig(filename, bbox_inches="tight", dpi=dpi)
    plt.close()  # Close the figure to free memory
    print(f"KAN architecture saved to {filename}")


def analyze_kan_feature_contributions(model, X_tensor, feature_names):
    model.eval()
    contributions = {}

    weights = model.state_dict()
    print("Weights shapes:")
    print("act_weights shape:", weights["act_fun.0.coef"].shape)
    print("sym_weights shape:", weights["symbolic_fun.0.affine"].shape)
    print("X_tensor shape:", X_tensor.shape)
    print("Number of features:", len(feature_names))

    act_weights = weights["act_fun.0.coef"].cpu().numpy()  # Non-linear contribution
    sym_weights = weights["symbolic_fun.0.affine"].cpu().numpy()  # Linear contribution

    for i in range(X_tensor.shape[1]):
        w_impact = np.abs(sym_weights[0, i, :]).mean()  # Adjusted for shape (1, 18, 4)
        a_impact = np.abs(act_weights[i, :, :]).mean()  # Adjusted for shape (18, 1, 7)
        contributions[feature_names[i]] = float(w_impact + a_impact)

    # Normalize and filter
    total = sum(contributions.values())
    contributions = {
        k: v / total for k, v in contributions.items() if v / total > 0.001
    }

    # Plot
    plt.figure(figsize=(12, 6))
    names = list(contributions.keys())
    values = list(contributions.values())
    plt.bar(names, values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Contributions Based on Model Weights")
    plt.tight_layout()
    plt.show()

    return dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))


def main():
    processor = KANDataProcessor1()
    visualizer = LiveVisualizer()
    X_data, y_target, config = processor.prepare_training_data(
        Path("Data/expression_data1.h5ad"), Path("Data/JUN_interactions.tsv"), "JUN"
    )
    related_genes = processor.get_related_genes("JUN")
    # Initialize model here
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
    epochs = 100
    # Create dataloaders
    train_loader, val_loader, test_loader = prepare_data(X_data, y_target)

    # Train model
    history = train_with_early_stopping(
        model, train_loader, val_loader, test_loader, optimizer, criterion, epochs
    )
    # plot_training_history(history)
    visualizer.plot_training_metrics(history)

    # Evaluate on test set
    test_loss, y_true, y_pred = evaluate_model(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f}")
    model.auto_symbolic()
    formula_tuple = model.symbolic_formula()
    print(formula_tuple[0])
    log_symbolic_formula(formula_tuple)  # Log the formula
    save_plot(model, "model_plot.png", dpi=1200)
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
    # Create input tensor
    X_tensor = torch.FloatTensor(X_data).to(device)

    # Load model with correct map_location
    checkpoint = torch.load("final_model.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Then move model to device
    model = model.to(device)
    X_tensor = X_tensor.to(device)
    contributions = analyze_kan_feature_contributions(model, X_tensor, related_genes)


if __name__ == "__main__":
    main()
