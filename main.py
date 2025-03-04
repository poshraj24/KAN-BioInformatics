import torch
import numpy as np
from pathlib import Path
from KAN_Train.data_processor import KANDataProcessor1
from KAN_Train.kan_model import *
import matplotlib.pyplot as plt
import warnings
from KAN_Train.visualizer import LiveVisualizer


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


def save_plot(model, filename="model_plot.png", dpi=3600):
    """Save the KAN architecture plot to a file."""
    model.plot(beta=10)
    plt.savefig(filename, bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"KAN architecture saved to {filename}")


def visualize_kan_node_importance(model, gene_names):
    """
    Visualize KAN's feature importance scores with gene names in their original node order.

    Args:
        model: A trained KAN model
        gene_names: List of gene names corresponding to input features

    Returns:
        Dictionary of gene importance scores in original order
    """
    # Set model to evaluation mode
    model.eval()

    # Get feature importance scores directly from the model
    feature_scores = model.feature_score.cpu().detach().numpy()

    # Map scores to gene names (maintaining original order)
    gene_importance = {}
    for i, gene in enumerate(gene_names):
        if i < len(feature_scores):
            gene_importance[gene] = float(feature_scores[i])

    # Normalize scores
    total = sum(abs(v) for v in gene_importance.values())
    if total > 0:
        gene_importance = {k: abs(v) / total for k, v in gene_importance.items()}

    # Create visualization
    plt.figure(figsize=(8, 6))
    genes = list(gene_importance.keys())
    scores = list(gene_importance.values())

    plt.bar(genes, scores)
    plt.xticks(rotation=90, ha="right")
    plt.title("Gene Importance Scores from KAN (Original Node Order)")
    plt.ylabel("Normalized Importance")
    plt.xlabel("Genes")
    plt.tight_layout()
    plt.show()

    return gene_importance


def main():
    processor = KANDataProcessor1()
    visualizer = LiveVisualizer()
    X_data, y_target, config = processor.prepare_training_data(
        Path("Data/expression_data1.h5ad"), Path("Data/JUN_interactions.tsv"), "JUN"
    )
    related_genes = processor.get_related_genes("JUN")

    # Initialize model
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
    save_plot(model, "model_plot.png", dpi=3600)
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

    # move model to device
    model = model.to(device)

    contributions = visualize_kan_node_importance(model, related_genes)


if __name__ == "__main__":
    main()
