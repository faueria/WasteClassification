"""
Model factory for CNN architecture instantiation.

Provides unified interface for loading pretrained models from timm library
with configurable classifier heads for waste classification.
"""
from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn
from torchinfo import summary

from configs.config import ModelConfig


# Architecture name mappings to timm model identifiers
ARCHITECTURE_REGISTRY = {
    "resnet50": "resnet50.a1_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "efficientnet_b3": "efficientnet_b3.ra2_in1k",
    "mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",
}


def create_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    drop_rate: float = 0.0,
) -> nn.Module:
    """
    Create a classification model with pretrained backbone.
    
    Args:
        architecture: Model architecture key (see ARCHITECTURE_REGISTRY).
        num_classes: Number of output classes.
        pretrained: Load ImageNet pretrained weights.
        freeze_backbone: If True, freeze all layers except classifier.
        drop_rate: Dropout rate before classifier (0.0 = no dropout).
        
    Returns:
        Configured PyTorch model ready for training.
        
    Raises:
        ValueError: If architecture is not supported.
    """
    if architecture not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported: {list(ARCHITECTURE_REGISTRY.keys())}"
        )
    
    timm_name = ARCHITECTURE_REGISTRY[architecture]
    
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    
    if freeze_backbone:
        _freeze_backbone(model, architecture)
    
    return model


def _freeze_backbone(model: nn.Module, architecture: str) -> None:
    """
    Freeze all parameters except the classifier head.
    
    Args:
        model: The model to modify in-place.
        architecture: Architecture key for identifying classifier layer.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier based on architecture
    classifier_attr = _get_classifier_attr(architecture)
    classifier = getattr(model, classifier_attr)
    
    for param in classifier.parameters():
        param.requires_grad = True


def _get_classifier_attr(architecture: str) -> str:
    """
    Get the attribute name for the classifier layer.
    
    Args:
        architecture: Model architecture key.
        
    Returns:
        String attribute name for accessing classifier.
    """
    # timm uses 'fc' for ResNet, 'classifier' for most others, 'head' for ConvNeXt
    classifier_mapping = {
        "resnet50": "fc",
        "convnext_tiny": "head",
        "efficientnet_b3": "classifier",
        "mobilenetv3_large": "classifier",
    }
    return classifier_mapping.get(architecture, "classifier")


def get_model_info(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224)) -> dict:
    """
    Extract model statistics including parameter count and FLOPs.
    
    Args:
        model: PyTorch model to analyze.
        input_size: Input tensor shape (batch, channels, height, width).
        
    Returns:
        Dict containing 'total_params', 'trainable_params', 'model_size_mb'.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB (float32 = 4 bytes per param)
    model_size_mb = total_params * 4 / (1024 ** 2)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "frozen_params": total_params - trainable_params,
    }


def print_model_summary(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
) -> None:
    """
    Print detailed model architecture summary.
    
    Args:
        model: Model to summarize.
        input_size: Input tensor shape.
        device: Device for summary computation.
    """
    if device:
        model = model.to(device)
    
    print(summary(
        model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=3,
        device=device,
    ))


def create_model_from_config(config: ModelConfig, num_classes: int) -> nn.Module:
    """
    Create model using configuration object.
    
    Args:
        config: ModelConfig with architecture settings.
        num_classes: Number of target classes.
        
    Returns:
        Configured model instance.
    """
    return create_model(
        architecture=config.architecture,
        num_classes=num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    )


def list_available_architectures() -> list:
    """Return list of supported architecture names."""
    return list(ARCHITECTURE_REGISTRY.keys())