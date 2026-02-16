#!/usr/bin/env python3
"""
Dataset verification and visualization script.

Validates TrashNet loading, displays class distributions, and shows sample images.
Run from project root: python -m scripts.verify_data
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import Config
from data import (
    load_and_split_data,
    create_dataloaders,
    get_base_transforms,
)
from data.transforms import IMAGENET_MEAN, IMAGENET_STD
from utils.device import print_device_info


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalization for visualization.
    
    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W].
        
    Returns:
        Denormalized tensor with values in [0, 1].
    """
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return torch.clamp(tensor * std + mean, 0, 1)


def plot_class_distribution(
    train_samples, val_samples, test_samples, class_names: list, save_path: Path
) -> None:
    """
    Create bar chart showing class distribution across splits.
    
    Args:
        train_samples: Training set (path, label) tuples.
        val_samples: Validation set tuples.
        test_samples: Test set tuples.
        class_names: List of class names.
        save_path: Output path for figure.
    """
    splits = {
        "Train": train_samples,
        "Validation": val_samples,
        "Test": test_samples,
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Class Distribution Across Data Splits", fontsize=14, fontweight="bold")
    
    colors = plt.cm.Set2.colors[:len(class_names)]
    
    for ax, (split_name, samples) in zip(axes, splits.items()):
        label_counts = Counter(label for _, label in samples)
        counts = [label_counts.get(i, 0) for i in range(len(class_names))]
        
        bars = ax.bar(class_names, counts, color=colors)
        ax.set_title(f"{split_name} (n={len(samples)})")
        ax.set_ylabel("Count")
        ax.set_xlabel("Class")
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.annotate(
                str(count),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
            )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved class distribution plot: {save_path}")


def plot_sample_images(
    dataloader, class_names: list, save_path: Path, num_samples: int = 16
) -> None:
    """
    Display grid of sample images from dataset.
    
    Args:
        dataloader: DataLoader to sample from.
        class_names: List of class names for titles.
        save_path: Output path for figure.
        num_samples: Number of images to display.
    """
    images, labels = next(iter(dataloader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Denormalize for display
    images = denormalize(images)
    
    # Create grid
    grid = make_grid(images, nrow=4, padding=2, normalize=False)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title("Sample Training Images", fontsize=14, fontweight="bold")
    
    # Add class labels
    label_str = ", ".join([class_names[l] for l in labels.tolist()])
    fig.text(0.5, 0.02, f"Labels: {label_str}", ha="center", fontsize=10)
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sample images: {save_path}")


def compute_dataset_statistics(dataloader) -> dict:
    """
    Compute channel-wise mean and std across dataset.
    
    Args:
        dataloader: DataLoader to iterate over.
        
    Returns:
        Dict with 'mean' and 'std' tensors.
    """
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_pixels = 0
    
    for images, _ in dataloader:
        channels_sum += images.sum(dim=[0, 2, 3])
        channels_squared_sum += (images ** 2).sum(dim=[0, 2, 3])
        num_pixels += images.shape[0] * images.shape[2] * images.shape[3]
    
    mean = channels_sum / num_pixels
    std = torch.sqrt(channels_squared_sum / num_pixels - mean ** 2)
    
    return {"mean": mean, "std": std}


def main():
    """Run full dataset verification pipeline."""
    print("=" * 60)
    print("WASTE CLASSIFICATION DATASET VERIFICATION")
    print("=" * 60)
    
    # Device info
    print("\n[1/5] Checking compute device...")
    print_device_info()
    
    # Load configuration
    print("\n[2/5] Loading configuration...")
    config = Config()
    print(f"  Dataset root: {config.data.root_dir}")
    print(f"  Classes: {config.data.class_names}")
    print(f"  Class mapping: {config.data.class_mapping}")
    print(f"  Split ratios: {config.data.train_ratio}/{config.data.val_ratio}/{config.data.test_ratio}")
    
    # Load and split data
    print("\n[3/5] Loading and splitting dataset...")
    try:
        train_samples, val_samples, test_samples = load_and_split_data(
            config.data, config.train.seed
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease download TrashNet dataset:")
        print("  wget https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip")
        print("  unzip dataset-resized.zip -d data/")
        sys.exit(1)
    
    # Print class distribution
    print("\n  Per-class counts:")
    for split_name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        counts = Counter(label for _, label in samples)
        dist = {config.data.class_names[k]: v for k, v in sorted(counts.items())}
        print(f"    {split_name}: {dist}")
    
    # Create dataloaders
    print("\n[4/5] Creating DataLoaders...")
    train_tf, eval_tf = get_base_transforms(config.data.image_size)
    train_loader, val_loader, test_loader = create_dataloaders(
        config, train_tf, eval_tf
    )
    
    # Verify batch loading
    batch, labels = next(iter(train_loader))
    print(f"  Batch shape: {batch.shape}")
    print(f"  Label dtype: {labels.dtype}")
    print(f"  Pixel range: [{batch.min():.2f}, {batch.max():.2f}]")
    
    # Generate visualizations
    print("\n[5/5] Generating visualizations...")
    output_dir = config.output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_class_distribution(
        train_samples, val_samples, test_samples,
        config.data.class_names,
        output_dir / "class_distribution.png"
    )
    
    plot_sample_images(
        train_loader,
        config.data.class_names,
        output_dir / "sample_images.png"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"  Total samples: {len(train_samples) + len(val_samples) + len(test_samples)}")
    print(f"  Classes: {len(config.data.class_names)}")
    print(f"  Figures saved to: {output_dir}")
    print("\nDataset is ready for training!")


if __name__ == "__main__":
    main()