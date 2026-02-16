"""
Evaluation metrics for classification performance.

Provides unified metric computation compatible with MPS tensors.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


@dataclass
class MetricsResult:
    """
    Container for classification metrics.
    
    Attributes:
        accuracy: Overall classification accuracy.
        f1_macro: Macro-averaged F1 score.
        f1_per_class: F1 score for each class.
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall.
        confusion_matrix: NxN confusion matrix.
        loss: Average loss value (if provided).
    """
    accuracy: float
    f1_macro: float
    f1_per_class: List[float]
    precision_macro: float
    recall_macro: float
    confusion_matrix: np.ndarray
    loss: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for logging."""
        result = {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
        }
        for i, f1 in enumerate(self.f1_per_class):
            result[f"f1_class_{i}"] = f1
        if self.loss is not None:
            result["loss"] = self.loss
        return result
    
    def __str__(self) -> str:
        """Format metrics for display."""
        lines = [
            f"Accuracy: {self.accuracy:.4f}",
            f"F1 Macro: {self.f1_macro:.4f}",
            f"Precision: {self.precision_macro:.4f}",
            f"Recall: {self.recall_macro:.4f}",
        ]
        if self.loss is not None:
            lines.insert(0, f"Loss: {self.loss:.4f}")
        return " | ".join(lines)


class MetricsCalculator:
    """
    Accumulates predictions and computes metrics at epoch end.
    
    Handles GPU/MPS tensor conversion automatically.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Args:
            num_classes: Number of classification targets.
            class_names: Optional names for classification report.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self) -> None:
        """Clear accumulated predictions for new epoch."""
        self.all_preds: List[int] = []
        self.all_labels: List[int] = []
        self.total_loss: float = 0.0
        self.num_batches: int = 0
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[float] = None,
    ) -> None:
        """
        Accumulate batch predictions.
        
        Args:
            predictions: Model output logits [B, C] or class indices [B].
            labels: Ground truth labels [B].
            loss: Batch loss value.
        """
        # Convert logits to class predictions if needed
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)
        
        # Move to CPU and convert to Python lists
        self.all_preds.extend(predictions.cpu().tolist())
        self.all_labels.extend(labels.cpu().tolist())
        
        if loss is not None:
            self.total_loss += loss
            self.num_batches += 1
    
    def compute(self) -> MetricsResult:
        """
        Compute all metrics from accumulated predictions.
        
        Returns:
            MetricsResult containing all computed metrics.
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0).tolist()
        precision = precision_score(labels, preds, average="macro", zero_division=0)
        recall = recall_score(labels, preds, average="macro", zero_division=0)
        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
        
        avg_loss = self.total_loss / max(self.num_batches, 1) if self.num_batches > 0 else None
        
        return MetricsResult(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_per_class=f1_per_class,
            precision_macro=precision,
            recall_macro=recall,
            confusion_matrix=cm,
            loss=avg_loss,
        )
    
    def get_classification_report(self) -> str:
        """Generate detailed sklearn classification report."""
        return classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.class_names,
            zero_division=0,
        )


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Compute inverse frequency weights for imbalanced classes.
    
    Args:
        labels: List of all training labels.
        num_classes: Total number of classes.
        
    Returns:
        Tensor of per-class weights for CrossEntropyLoss.
    """
    counts = torch.zeros(num_classes)
    for label in labels:
        counts[label] += 1
    
    # Inverse frequency weighting
    weights = 1.0 / counts.clamp(min=1)
    weights = weights / weights.sum() * num_classes
    
    return weights