#!/usr/bin/env python3
"""
Baseline training script for waste classification.

Trains ResNet-50 with standard augmentation as the project baseline.
All team members should run this first to establish ground truth.

Usage:
    python -m scripts.train_baseline
    python -m scripts.train_baseline --architecture convnext_tiny --epochs 30
"""
import argparse
import json
import sys
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config

from data import create_dataloaders, get_augmentation_pipeline
from models import create_model, get_model_info
from utils import (
    get_device,
    print_device_info,
    Trainer,
    MetricsCalculator,
)
from utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_per_class_metrics,
)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train waste classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet50",
        choices=["resnet50", "convnext_tiny", "efficientnet_b3", "mobilenetv3_large"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone, train only classifier",
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Data arguments
    parser.add_argument(
        "--augmentation",
        type=str,
        default="standard",
        choices=["base", "standard", "trivialaugment"],
        help="Augmentation strategy",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    """Execute baseline training pipeline."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.experiment_name or f"{args.architecture}_{args.augmentation}_{timestamp}"
    
    print("=" * 70)
    print("WASTE CLASSIFICATION - BASELINE TRAINING")
    print("=" * 70)
    print(f"Experiment: {exp_name}")
    print(f"Architecture: {args.architecture}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    
    # Setup device
    print("\n[1/6] Setting up device...")
    device = get_device()
    print_device_info()
    
    # Setup output directories
    exp_dir = args.output_dir / exp_name
    checkpoint_dir = exp_dir / "checkpoints"
    figures_dir = exp_dir / "figures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    print("\n[2/6] Loading configuration...")
    config = Config()
    config.train.batch_size = args.batch_size
    config.train.epochs = args.epochs
    config.train.learning_rate = args.lr
    config.train.weight_decay = args.weight_decay
    config.train.seed = args.seed
    config.data.num_workers = args.num_workers
    
    print(f"  Classes: {config.data.class_names}")
    print(f"  Num classes: {config.data.num_classes}")
    
    # Create data loaders
    print("\n[3/6] Loading dataset...")
    train_transform, eval_transform = get_augmentation_pipeline(
        args.augmentation, config.data.image_size
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        config, train_transform, eval_transform,
        use_weighted_sampler=True
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print("\n[4/6] Creating model...")
    model = create_model(
        architecture=args.architecture,
        num_classes=config.data.num_classes,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    )
    
    model_info = get_model_info(model)
    print(f"  Total params: {model_info['total_params']:,}")
    print(f"  Trainable params: {model_info['trainable_params']:,}")
    print(f"  Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Setup training components
    print("\n[5/6] Configuring training...")
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_classes=config.data.num_classes,
        class_names=config.data.class_names,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=args.patience,
    )
    
    # Train model
    print("\n[6/6] Starting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        verbose=True,
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    # Load best model
    best_checkpoint = checkpoint_dir / "best_model.pth"
    if best_checkpoint.exists():
        trainer.load_checkpoint(best_checkpoint)
        print(f"Loaded best model from epoch {trainer.best_epoch}")
    
    test_metrics = trainer.validate(test_loader)
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics.accuracy:.4f}")
    print(f"  F1 Macro: {test_metrics.f1_macro:.4f}")
    print(f"  Precision: {test_metrics.precision_macro:.4f}")
    print(f"  Recall: {test_metrics.recall_macro:.4f}")
    
    print(f"\nPer-class F1 scores:")
    for name, f1 in zip(config.data.class_names, test_metrics.f1_per_class):
        print(f"  {name}: {f1:.4f}")
    
    # Generate visualizations
    print("\n" + "-" * 70)
    print("Generating visualizations...")
    
    plot_training_curves(
        history,
        save_path=figures_dir / "training_curves.png",
        title=f"Training History - {args.architecture}",
    )
    
    plot_confusion_matrix(
        test_metrics.confusion_matrix,
        config.data.class_names,
        save_path=figures_dir / "confusion_matrix.png",
        title=f"Test Confusion Matrix - {args.architecture}",
    )
    
    plot_per_class_metrics(
        test_metrics.f1_per_class,
        config.data.class_names,
        save_path=figures_dir / "per_class_f1.png",
        title=f"Per-Class F1 Scores - {args.architecture}",
    )
    
    # Save experiment results
    results = {
        "experiment_name": exp_name,
        "architecture": args.architecture,
        "augmentation": args.augmentation,
        "epochs_trained": len(history.train_loss),
        "best_epoch": trainer.best_epoch,
        "test_metrics": test_metrics.to_dict(),
        "model_info": model_info,
        "args": vars(args),
        "history": history.to_dict(),
    }
    
    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results: {results_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Experiment directory: {exp_dir}")
    print(f"Best validation F1: {trainer.best_val_f1:.4f} (epoch {trainer.best_epoch})")
    print(f"Test F1: {test_metrics.f1_macro:.4f}")
    print(f"Test Accuracy: {test_metrics.accuracy:.4f}")


if __name__ == "__main__":
    main()