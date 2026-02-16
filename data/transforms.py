"""
Image transformation pipelines for training and evaluation.

Provides baseline and advanced augmentation strategies for Person B's experiments.
"""
from typing import Tuple

from torchvision import transforms
from torchvision.transforms import v2

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_base_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Baseline transforms with minimal augmentation.
    
    Args:
        image_size: Target image dimension (square).
        
    Returns:
        Tuple of (train_transform, eval_transform).
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    return train_transform, eval_transform


def get_standard_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Standard augmentation with color jitter and random crop.
    
    Args:
        image_size: Target image dimension.
        
    Returns:
        Tuple of (train_transform, eval_transform).
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    return train_transform, eval_transform


def get_trivialaugment_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    TrivialAugmentWide for automated augmentation policy.
    
    Reference: Müller & Hutter, "TrivialAugment" (ICCV 2021).
    
    Args:
        image_size: Target image dimension.
        
    Returns:
        Tuple of (train_transform, eval_transform).
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    return train_transform, eval_transform


class MixUpCutMixCollator:
    """
    Batch-level MixUp/CutMix augmentation collator.
    
    Randomly applies either MixUp or CutMix to each batch with 50% probability each.
    Used as collate_fn in DataLoader for Person B's augmentation experiments.
    
    Attributes:
        num_classes: Number of target classes for label smoothing.
        mixup_alpha: Beta distribution parameter for MixUp.
        cutmix_alpha: Beta distribution parameter for CutMix.
    """
    
    def __init__(self, num_classes: int, mixup_alpha: float = 0.4, cutmix_alpha: float = 1.0):
        """
        Args:
            num_classes: Number of classification targets.
            mixup_alpha: MixUp interpolation strength (higher = more mixing).
            cutmix_alpha: CutMix patch size parameter.
        """
        self.num_classes = num_classes
        self.transform = v2.RandomChoice([
            v2.MixUp(num_classes=num_classes, alpha=mixup_alpha),
            v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha),
        ])
    
    def __call__(self, batch):
        """
        Apply MixUp or CutMix to collated batch.
        
        Args:
            batch: List of (image, label) tuples from Dataset.
            
        Returns:
            Tuple of (mixed_images, soft_labels) tensors.
        """
        import torch
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return self.transform(images, labels)


def get_augmentation_pipeline(
    augmentation_type: str,
    image_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Factory function for augmentation pipelines.
    
    Args:
        augmentation_type: One of 'base', 'standard', 'trivialaugment'.
        image_size: Target image dimension.
        
    Returns:
        Tuple of (train_transform, eval_transform).
        
    Raises:
        ValueError: If augmentation_type is not recognized.
    """
    pipelines = {
        "base": get_base_transforms,
        "standard": get_standard_transforms,
        "trivialaugment": get_trivialaugment_transforms,
    }
    
    if augmentation_type not in pipelines:
        raise ValueError(
            f"Unknown augmentation type: {augmentation_type}. "
            f"Choose from: {list(pipelines.keys())}"
        )
    
    return pipelines[augmentation_type](image_size)