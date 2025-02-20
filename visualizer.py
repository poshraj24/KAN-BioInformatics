import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import torch


class LiveVisualizer:

    def plot_training_metrics(self, metrics_history, save_path="visualizations"):
        save_path = os.path.abspath(save_path)
        os.makedirs(save_path, exist_ok=True)
        sns.set_style("darkgrid")

        # Plot only if data exists
        if all(
            key in metrics_history and len(metrics_history[key]) > 0
            for key in ["train_loss", "val_loss", "test_loss"]
        ):
            self._plot_losses(metrics_history, save_path)

        if all(
            key in metrics_history and len(metrics_history[key]) > 0
            for key in ["train_r2", "val_r2", "test_r2"]
        ):
            self._plot_r2_scores(metrics_history, save_path)

        if (
            "learning_rate" in metrics_history
            and len(metrics_history["learning_rate"]) > 0
        ):
            self._plot_learning_rate(metrics_history, save_path)

        if all(
            key in metrics_history and len(metrics_history[key]) > 0
            for key in ["train_loss", "val_loss", "train_r2", "val_r2"]
        ):
            self._plot_training_progress(metrics_history, save_path)

        if all(
            key in metrics_history and len(metrics_history[key]) > 0
            for key in ["y_true", "y_pred"]
        ):
            self._plot_predictions(metrics_history, save_path)

    def _plot_losses(self, metrics_history, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_history["train_loss"], label="Training Loss", color="blue")
        plt.plot(metrics_history["val_loss"], label="Validation Loss", color="red")
        plt.plot(metrics_history["test_loss"], label="Test Loss", color="green")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.title("Training, Validation, and Test Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/loss_curves.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_r2_scores(self, metrics_history, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_history["train_r2"], label="Training R²", color="blue")
        plt.plot(metrics_history["val_r2"], label="Validation R²", color="red")
        plt.plot(metrics_history["test_r2"], label="Test R²", color="green")
        plt.xlabel("Epochs")
        plt.ylabel("R² Score")
        plt.title("R² Score Progression")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/r2_scores.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_learning_rate(self, metrics_history, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_history["learning_rate"], color="purple")
        plt.xlabel("Steps")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/learning_rate.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_training_progress(self, metrics_history, save_path):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.plot(metrics_history["train_loss"], label="Training", color="blue")
        ax1.plot(metrics_history["val_loss"], label="Validation", color="red")
        ax1.set_ylabel("Loss (MSE)")
        ax1.set_title("Training Progress")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(metrics_history["train_r2"], label="Training", color="blue")
        ax2.plot(metrics_history["val_r2"], label="Validation", color="red")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("R² Score")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"{save_path}/training_progress.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_predictions(self, metrics_history, save_path):
        plt.figure(figsize=(10, 6))
        plt.scatter(metrics_history["y_true"], metrics_history["y_pred"], alpha=0.5)
        plt.plot(
            [min(metrics_history["y_true"]), max(metrics_history["y_true"])],
            [min(metrics_history["y_true"]), max(metrics_history["y_true"])],
            "r--",
            label="Perfect Prediction",
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Prediction vs Actual Values")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/prediction_scatter.pdf", dpi=300, bbox_inches="tight")
        plt.close()
