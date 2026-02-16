"""
Training visualization utilities.

Generates publication-quality figures for training curves and confusion matrices.
"""
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.trainer import TrainingHistory


def plot_training_curves(
    history: TrainingHistory,
    save_path: Optional[Path] = None,
    title: str = "Training History",
) -> None:
    """
    Plot loss and accuracy curves for training and validation.
    
    Args:
        history: TrainingHistory from Trainer.fit().
        save_path: Optional path to save figure.
        title: Figure title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history.train_loss) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history.train_loss, "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, history.val_loss, "r-", label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, history.train_acc, "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, history.val_acc, "r-", label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 curves
    axes[2].plot(epochs, history.train_f1, "b-", label="Train", linewidth=2)
    axes[2].plot(epochs, history.val_f1, "r-", label="Validation", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Macro F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves: {save_path}")
    
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix array (n_classes x n_classes).
        class_names: List of class names for axis labels.
        save_path: Optional path to save figure.
        title: Figure title.
        normalize: If True, show percentages instead of counts.
    """
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix: {save_path}")
    
    plt.close()


def plot_per_class_metrics(
    f1_scores: List[float],
    class_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "Per-Class F1 Scores",
) -> None:
    """
    Bar chart of F1 scores per class.
    
    Args:
        f1_scores: List of F1 scores for each class.
        class_names: Corresponding class names.
        save_path: Optional path to save figure.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = plt.cm.Set2.colors[:len(class_names)]
    bars = ax.bar(class_names, f1_scores, color=colors, edgecolor="black")
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        ax.annotate(
            f"{score:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=np.mean(f1_scores), color="red", linestyle="--", label=f"Mean: {np.mean(f1_scores):.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved per-class metrics: {save_path}")
    
    plt.close()


def plot_learning_rate_schedule(
    learning_rates: List[float],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot learning rate across epochs.
    
    Args:
        learning_rates: LR values per epoch.
        save_path: Optional path to save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    epochs = range(1, len(learning_rates) + 1)
    ax.plot(epochs, learning_rates, "b-", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved LR schedule: {save_path}")
    
    plt.close()