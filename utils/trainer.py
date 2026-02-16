"""
Training orchestration with validation, checkpointing, and early stopping.

Provides a reusable Trainer class for all experiments.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy.core.multiarray

from utils.device import clear_memory_cache, get_memory_usage
from utils.metrics import MetricsCalculator, MetricsResult


@dataclass
class TrainingHistory:
    """
    Records training metrics across epochs.
    
    Attributes:
        train_loss: Training loss per epoch.
        val_loss: Validation loss per epoch.
        train_acc: Training accuracy per epoch.
        val_acc: Validation accuracy per epoch.
        train_f1: Training F1 per epoch.
        val_f1: Validation F1 per epoch.
        learning_rates: LR per epoch.
        epoch_times: Seconds per epoch.
    """
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    train_f1: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary for serialization."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "train_f1": self.train_f1,
            "val_f1": self.val_f1,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
        }


class EarlyStopping:
    """
    Monitors validation metric and stops training when improvement stalls.
    
    Attributes:
        patience: Epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' for loss, 'max' for accuracy/F1.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
    ):
        """
        Args:
            patience: Number of epochs without improvement before stopping.
            min_delta: Minimum improvement threshold.
            mode: 'min' to minimize metric, 'max' to maximize.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current validation metric value.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        improved = (
            (self.mode == "max" and value > self.best_value + self.min_delta) or
            (self.mode == "min" and value < self.best_value - self.min_delta)
        )
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """
    Orchestrates model training with validation, logging, and checkpointing.
    
    Supports MPS acceleration, mixed precision (where available), and
    customizable loss functions for Person B's experiments.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_dir: Optional[Path] = None,
        early_stopping_patience: int = 10,
    ):
        """
        Args:
            model: PyTorch model to train.
            optimizer: Optimizer instance.
            criterion: Loss function.
            device: Compute device (cpu/cuda/mps).
            num_classes: Number of target classes.
            class_names: Names for metrics reporting.
            scheduler: Optional LR scheduler.
            checkpoint_dir: Directory for saving checkpoints.
            early_stopping_patience: Epochs without improvement before stopping.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        
        self.metrics_calc = MetricsCalculator(num_classes, class_names)
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")
        self.history = TrainingHistory()
        
        self.best_val_f1: float = 0.0
        self.best_epoch: int = 0
        
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, dataloader: DataLoader) -> MetricsResult:
        """
        Execute one training epoch.
        
        Args:
            dataloader: Training data loader.
            
        Returns:
            MetricsResult for the epoch.
        """
        self.model.train()
        self.metrics_calc.reset()
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            self.metrics_calc.update(outputs.detach(), labels, loss.item())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return self.metrics_calc.compute()
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> MetricsResult:
        """
        Evaluate model on validation/test set.
        
        Args:
            dataloader: Validation or test data loader.
            
        Returns:
            MetricsResult for the evaluation.
        """
        self.model.eval()
        self.metrics_calc.reset()
        
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.metrics_calc.update(outputs, labels, loss.item())
        
        return self.metrics_calc.compute()
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Full training loop with validation and early stopping.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Maximum number of epochs.
            verbose: Print progress to stdout.
            
        Returns:
            TrainingHistory with all recorded metrics.
        """
        if verbose:
            print(f"\nStarting training for {epochs} epochs on {self.device}")
            print(f"Early stopping patience: {self.early_stopping.patience}")
            print("-" * 60)
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            epoch_time = time.time() - epoch_start
            self._record_epoch(train_metrics, val_metrics, current_lr, epoch_time)
            
            # Print progress
            if verbose:
                self._print_epoch(epoch, epochs, train_metrics, val_metrics, epoch_time)
            
            # Save best model
            if val_metrics.f1_macro > self.best_val_f1:
                self.best_val_f1 = val_metrics.f1_macro
                self.best_epoch = epoch
                if self.checkpoint_dir:
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
            
            # Early stopping check
            if self.early_stopping(val_metrics.f1_macro):
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
                break
            
            # Memory cleanup every 5 epochs
            if epoch % 5 == 0:
                clear_memory_cache(self.device)
        
        if verbose:
            print("-" * 60)
            print(f"Training complete. Best val F1: {self.best_val_f1:.4f}")
        
        return self.history
    
    def _record_epoch(
        self,
        train: MetricsResult,
        val: MetricsResult,
        lr: float,
        time_sec: float,
    ) -> None:
        """Append epoch metrics to history."""
        self.history.train_loss.append(train.loss or 0)
        self.history.val_loss.append(val.loss or 0)
        self.history.train_acc.append(train.accuracy)
        self.history.val_acc.append(val.accuracy)
        self.history.train_f1.append(train.f1_macro)
        self.history.val_f1.append(val.f1_macro)
        self.history.learning_rates.append(lr)
        self.history.epoch_times.append(time_sec)
    
    def _print_epoch(
        self,
        epoch: int,
        total: int,
        train: MetricsResult,
        val: MetricsResult,
        time_sec: float,
    ) -> None:
        """Format and print epoch summary."""
        mem = get_memory_usage(self.device)
        mem_str = f" | Mem: {mem:.1f}GB" if mem > 0 else ""
        
        print(
            f"Epoch {epoch:3d}/{total} | "
            f"Train Loss: {train.loss:.4f} Acc: {train.accuracy:.4f} | "
            f"Val Loss: {val.loss:.4f} Acc: {val.accuracy:.4f} F1: {val.f1_macro:.4f} | "
            f"Time: {time_sec:.1f}s{mem_str}"
        )
    
    def _save_checkpoint(
        self,
        epoch: int,
        metrics: MetricsResult,
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint to disk."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics.to_dict(),
            "history": self.history.to_dict(),
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> int:
        """
        Load model from checkpoint file.
        
        Args:
            path: Path to checkpoint .pth file.
            
        Returns:
            Epoch number from checkpoint.
        """
        torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint["epoch"]