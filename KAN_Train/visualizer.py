import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import torch


class LiveVisualizer:

    @staticmethod
    def plot_training_metrics(history):
        epochs = np.array(range(1, len(history) + 1))
        metrics = [
            np.array([x.cpu().numpy() if torch.is_tensor(x) else x for x in m])
            for m in zip(*history)
        ]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss Plot
        ax1.plot(epochs, metrics[0], "b-", label="Train Loss")
        ax1.plot(epochs, metrics[1], "r-", label="Val Loss")
        ax1.plot(epochs, metrics[2], "g-", label="Test Loss")
        ax1.set_title("Loss")
        ax1.legend()

        # R² Plot
        ax2.plot(epochs, metrics[3], "r-", label="Validation R²")
        ax2.set_title("R² Score")
        ax2.legend()

        # RMSE Plot
        ax3.plot(epochs, metrics[4], "r-", label="Validation RMSE")
        ax3.set_title("RMSE")
        ax3.legend()

        # MAE Plot
        ax4.plot(epochs, metrics[5], "r-", label="Validation MAE")
        ax4.set_title("MAE")
        ax4.legend()

        plt.tight_layout()
        plt.savefig("training_metrics.png")
        plt.show()
