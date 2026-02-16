"""
Central configuration for waste classification project.

Defines hyperparameters, paths, and class mappings used across all modules.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

@dataclass
class DataConfig:
    """Dataset configuration parameters."""
    # root_dir: Path = Path("C:/Users/Nihal Sandadi/Desktop/computer vision/Final Project/waste-classification-main/data/dataset-resized")
    root_dir: Path = Path("C:/Users/Nihal Sandadi/Downloads/archive/new-dataset-trash-type-v2")
    # root_dir: Path = Path("data/dataset-resized")
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    image_size: int = 248
    num_workers: int = 4
    pin_memory: bool = False  # Not beneficial for MPS unified memory
    
    # 6-class to 3-class mapping (TrashNet -> Recyclable/Organic/Non-Recyclable)
    class_mapping: Dict[str, str] = field(default_factory=lambda: {
        "cardboard": "cardboard",
        #"ewaste": "ewaste",
        "glass": "glass",
        "metal": "metal", 
        "paper": "paper",
        "plastic": "plastic",
        #"textile": "textile",
        "trash": "trash",
        #"organic": "organic",
    })
    
    @property
    def class_names(self) -> List[str]:
        return ["cardboard", 
                #"ewaste"
                "glass", 
                "metal", 
                "paper", 
                "plastic", 
                #"textile"
                "trash"#,
                #"organic"
                ]
    
    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    seed: int = 42
    early_stopping_patience: int = 10
    

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    drop_rate: float = 0.2
    
    # Supported architectures for Person A's comparison
    SUPPORTED_ARCHS: List[str] = field(default_factory=lambda: [
        "resnet50",
        "convnext_tiny",
        "efficientnet_b3",
        "mobilenetv3_large",
    ])


@dataclass
class Config:
    """Master configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: Path = Path("outputs")
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)