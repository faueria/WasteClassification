# Waste Classification CNN

A deep learning system for classifying waste images into recyclable and non-recyclable categories using transfer learning with modern CNN architectures.

## Project Overview

This project implements a waste classification pipeline with three complementary research contributions:

| Component | Owner | Focus |
|-----------|-------|-------|
| Architecture Comparison | Person A | ResNet-50, ConvNeXt-Tiny, EfficientNet-B3, MobileNetV3 |
| Augmentation & Class Imbalance | Person B | TrivialAugment, MixUp/CutMix, Focal Loss |
| Interpretability & Deployment | Person C | Grad-CAM++, Temperature Scaling, Gradio Demo |

**Target Performance:** 95%+ accuracy on TrashNet dataset with interpretable predictions.

## Installation

### Prerequisites

- Python 3.11+
- macOS with M1/M2 Pro (MPS acceleration) or NVIDIA GPU (CUDA)
- 8GB+ RAM

### Setup

```bash
# Clone repository
git clone <repository-url>
cd waste-classification

# Create conda environment
conda create -n waste-cnn python=3.11 -y
conda activate waste-cnn

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the TrashNet dataset:

```bash
# Option 1: Direct download
wget https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip
unzip dataset-resized.zip -d data/

# Option 2: Manual download
# Visit https://github.com/garythung/trashnet and download dataset-resized.zip
# Extract to data/dataset-resized/
```

Expected structure:
```
data/
└── dataset-resized/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

## Project Structure

```
waste_classification/
├── __init__.py
├── configs/
│   ├── __init__.py
│   └── config.py              # Central configuration (data, training, model)
├── data/
│   ├── __init__.py
│   ├── dataset.py             # WasteDataset class, data loading utilities
│   └── transforms.py          # Augmentation pipelines (base, standard, trivialaugment)
├── models/
│   ├── __init__.py
│   └── factory.py             # Model factory with timm integration
├── utils/
│   ├── __init__.py
│   ├── device.py              # MPS/CUDA device management
│   ├── metrics.py             # MetricsCalculator, F1, accuracy, confusion matrix
│   ├── trainer.py             # Trainer class with early stopping
│   └── visualization.py       # Training curves, confusion matrix plots
└── scripts/
    ├── __init__.py
    ├── verify_data.py         # Dataset verification and visualization
    └── train_baseline.py      # Main training script
```

## Quick Start

### 1. Verify Dataset

```bash
python -m scripts.verify_data
```

This validates the dataset, displays class distributions, and generates sample visualizations in `outputs/figures/`.

### 2. Train Baseline Model

```bash
# Quick test run (5 epochs)
python -m scripts.train_baseline --epochs 5 --experiment-name quick_test

# Full baseline training
python -m scripts.train_baseline \
    --architecture resnet50 \
    --augmentation standard \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4
```

### 3. Compare Architectures

```bash
# Train different architectures (Person A's experiments)
for arch in resnet50 convnext_tiny efficientnet_b3 mobilenetv3_large; do
    python -m scripts.train_baseline \
        --architecture $arch \
        --augmentation standard \
        --epochs 50 \
        --experiment-name ${arch}_baseline
done
```

## Training Script Options

```
usage: train_baseline.py [-h] [--architecture {resnet50,convnext_tiny,efficientnet_b3,mobilenetv3_large}]
                         [--freeze-backbone] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                         [--lr LR] [--weight-decay WEIGHT_DECAY] [--patience PATIENCE]
                         [--augmentation {base,standard,trivialaugment}]
                         [--num-workers NUM_WORKERS] [--output-dir OUTPUT_DIR]
                         [--experiment-name EXPERIMENT_NAME] [--seed SEED]

Arguments:
  --architecture        Model architecture (default: resnet50)
  --freeze-backbone     Freeze backbone, train only classifier head
  --epochs              Number of training epochs (default: 50)
  --batch-size          Batch size (default: 32)
  --lr                  Learning rate (default: 1e-4)
  --weight-decay        AdamW weight decay (default: 0.01)
  --patience            Early stopping patience (default: 10)
  --augmentation        Augmentation strategy (default: standard)
  --num-workers         DataLoader workers (default: 4)
  --output-dir          Output directory (default: outputs/)
  --experiment-name     Custom experiment name
  --seed                Random seed (default: 42)
```

## Configuration

### Class Mapping

TrashNet's 6 classes are mapped to 2 categories:

| Original Class | Target Class |
|---------------|--------------|
| cardboard, glass, metal, paper, plastic | recyclable |
| trash | non_recyclable |

### Default Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image Size | 224×224 | Standard ImageNet input |
| Batch Size | 32 | Safe for 16GB unified memory |
| Learning Rate | 1e-4 | With cosine annealing to 1e-6 |
| Weight Decay | 0.01 | AdamW regularization |
| Epochs | 50 | With early stopping (patience=10) |
| Train/Val/Test Split | 70/15/15 | Stratified by class |

### Supported Architectures

| Architecture | Parameters | ImageNet Top-1 | Notes |
|-------------|-----------|----------------|-------|
| ResNet-50 | 25.6M | 76.3% | Well-understood baseline |
| ConvNeXt-Tiny | 28M | 82.1% | Best transfer learning |
| EfficientNet-B3 | 12M | 81.6% | Parameter efficient |
| MobileNetV3-Large | 5.4M | 75.2% | Deployment optimized |

## Output Structure

Each training run creates:

```
outputs/<experiment_name>/
├── checkpoints/
│   └── best_model.pth          # Best validation F1 checkpoint
├── figures/
│   ├── training_curves.png     # Loss, accuracy, F1 over epochs
│   ├── confusion_matrix.png    # Test set confusion matrix
│   └── per_class_f1.png        # Per-class F1 bar chart
└── results.json                # Full metrics and configuration
```

## API Usage

### Create Model

```python
from models import create_model, get_model_info

model = create_model(
    architecture="convnext_tiny",
    num_classes=2,
    pretrained=True,
    freeze_backbone=False,
)

info = get_model_info(model)
print(f"Parameters: {info['total_params']:,}")
```

### Load Dataset

```python
from configs import Config
from data import create_dataloaders, get_augmentation_pipeline

config = Config()
train_tf, eval_tf = get_augmentation_pipeline("standard", image_size=224)
train_loader, val_loader, test_loader = create_dataloaders(config, train_tf, eval_tf)
```

### Custom Training Loop

```python
from utils import Trainer, get_device

device = get_device()
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_classes=2,
    class_names=["non_recyclable", "recyclable"],
    checkpoint_dir=Path("outputs/my_experiment/checkpoints"),
)

history = trainer.fit(train_loader, val_loader, epochs=50)
```

## Hardware Optimization

### Apple Silicon (M1/M2 Pro)

The codebase is optimized for MPS acceleration:

- Automatic MPS device detection
- CPU fallback enabled for unsupported operations
- Memory cache clearing every 5 epochs
- `pin_memory=False` (not beneficial with unified memory)

### NVIDIA GPU

CUDA is automatically detected and used when available. For multi-GPU training, additional configuration is required (not implemented in baseline).

## Metrics

All experiments report:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| F1 Macro | Class-balanced F1 score |
| F1 Per-Class | Individual class F1 scores |
| Precision | Macro-averaged precision |
| Recall | Macro-averaged recall |
| Confusion Matrix | True vs predicted label counts |

## References

1. Thung & Yang, "Classification of Trash for Recyclability Status," Stanford CS229, 2016
2. Mao et al., "Recycling waste classification using optimized CNN," Resources, Conservation and Recycling, 2021
3. Chattopadhyay et al., "Grad-CAM++: Generalized gradient-based visual explanations," WACV 2018
4. Guo et al., "On calibration of modern neural networks," ICML 2017
5. Yun et al., "CutMix: Regularization strategy to train strong classifiers," ICCV 2019

## License

MIT License - See LICENSE file for details.

## Team

- **Person A** - Architecture comparison and optimization
- **Person B** - Data augmentation and class imbalance handling  
- **Person C** - Model interpretability and deployment
