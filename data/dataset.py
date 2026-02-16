"""
TrashNet dataset loader with 6-to-3 class mapping.

Provides stratified splitting and class-balanced sampling utilities.
"""
import random
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from configs.config import Config, DataConfig


class WasteDataset(Dataset):
    """
    PyTorch Dataset for waste classification with configurable class mapping.
    
    Attributes:
        samples: List of (image_path, label_index) tuples.
        transform: Optional torchvision transforms to apply.
        class_names: List of target class names.
    """
    
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        class_names: List[str],
        transform=None,
    ):
        """
        Args:
            samples: Pre-split list of (path, label) tuples.
            class_names: Ordered list of class names for label mapping.
            transform: Image transforms to apply.
        """
        self.samples = samples
        self.class_names = class_names
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and transform a single sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (transformed_image, label_index).
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse frequency weights for class imbalance handling.
        
        Returns:
            Tensor of per-class weights (higher for minority classes).
        """
        label_counts = torch.zeros(len(self.class_names))
        for _, label in self.samples:
            label_counts[label] += 1
        
        weights = 1.0 / label_counts
        weights = weights / weights.sum() * len(self.class_names)
        return weights
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights for WeightedRandomSampler.
        
        Returns:
            Tensor of weight per sample based on class frequency.
        """
        class_weights = self.get_class_weights()
        return torch.tensor([class_weights[label] for _, label in self.samples])


def load_and_split_data(
    config: DataConfig,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Load TrashNet images and create stratified train/val/test splits.
    
    Applies 6-to-3 class mapping defined in config.
    
    Args:
        config: DataConfig with paths and split ratios.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples).
        
    Raises:
        FileNotFoundError: If dataset directory doesn't exist.
    """
    root = Path(config.root_dir)
    # Debug current state
    print(f"Config root_dir: '{config.root_dir}'")
    print(f"Current working directory: '{os.getcwd()}'")

    # Check if it's a relative path
    root = Path(config.root_dir)
    print(f"Root path object: {root}")
    print(f"Is absolute? {root.is_absolute()}")
    print(f"Absolute path: {root.absolute()}")

    # Check parent directories
    print(f"Parent exists? {root.parent.exists()}")
    print(f"Parent of parent exists? {root.parent.parent.exists() if root.parent else 'No parent'}")

    # List contents of data directory if it exists
    data_dir = Path("data")
    if data_dir.exists():
        print(f"Contents of data directory: {list(data_dir.iterdir())}")
    else:
        print("'data' directory doesn't exist!")

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {root.absolute()}. "
            f"Current working directory: {os.getcwd()}"
        )
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {root}. "
            "Download TrashNet and extract to data/dataset-resized/"
        )
    
    # Build class name to index mapping
    class_to_idx = {name: idx for idx, name in enumerate(config.class_names)}
    
    # Collect all samples with remapped labels
    all_samples: List[Tuple[Path, int]] = []
    
    for original_class, target_class in config.class_mapping.items():
        class_dir = root / original_class
        if not class_dir.exists():
            print(f"Warning: Directory {class_dir} not found, skipping.")
            continue
        
        target_idx = class_to_idx[target_class]
        
        valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        for img_path in class_dir.iterdir():
            if img_path.suffix in valid_extensions:
                all_samples.append((img_path, target_idx))
            else:
                # Optional: print what files are being skipped
                pass
    
    if not all_samples:
        raise FileNotFoundError(f"No images found in {root}")
    
    print(f"Total samples loaded: {len(all_samples)}")
    
    # Extract labels for stratification
    labels = [label for _, label in all_samples]
    
    # First split: train vs (val + test)
    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        all_samples,
        labels,
        test_size=(config.val_ratio + config.test_ratio),
        stratify=labels,
        random_state=seed,
    )
    
    # Second split: val vs test
    val_ratio_adjusted = config.val_ratio / (config.val_ratio + config.test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_labels,
        random_state=seed,
    )
    
    # Print split statistics
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    return train_samples, val_samples, test_samples


def create_dataloaders(
    config: Config,
    train_transform=None,
    eval_transform=None,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with optional class balancing.
    
    Args:
        config: Master configuration object.
        train_transform: Transforms for training data.
        eval_transform: Transforms for validation/test data.
        use_weighted_sampler: If True, use WeightedRandomSampler for training.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_samples, val_samples, test_samples = load_and_split_data(
        config.data, config.train.seed
    )
    
    train_dataset = WasteDataset(
        train_samples, config.data.class_names, train_transform
    )
    val_dataset = WasteDataset(
        val_samples, config.data.class_names, eval_transform
    )
    test_dataset = WasteDataset(
        test_samples, config.data.class_names, eval_transform
    )
    
    # Configure sampler for class imbalance
    train_sampler = None
    train_shuffle = True
    
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = False  # Sampler handles randomization
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    return train_loader, val_loader, test_loader