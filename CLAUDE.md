# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GEDI-ORDER is a Convolutional Neural Network (CNN) project for binary cell classification (alive/dead) using the Genetically Encoded Death Indicator (GEDI) biomarker. The system combines deep learning with gradient-based visualization (Grad-CAM) to provide interpretable predictions of cell vitality based on morphological features.

## Core Architecture

The codebase is organized into functional modules:

- **main/**: Training orchestration and model creation
- **deploy/**: Model deployment and inference pipeline
- **preprocessing/**: Data pipeline (TFRecord creation, data generators)
- **models/**: CNN architecture definitions (VGG16, VGG19, ResNet50)
- **activationmap/**: Grad-CAM visualization and gradient analysis
- **ops/**: Core TensorFlow operations and processing utilities
- **utils/**: General utilities and helper functions
- **vis/**: Plotting and visualization tools

### Key Components

- **param_gedi.py**: Central configuration system with machine-specific paths and hyperparameters
- **models/model.py**: CNN class with multiple architecture support
- **preprocessing/datagenerator.py**: TensorFlow data pipeline with augmentation
- **activationmap/gradcam.py**: Grad-CAM implementation for model interpretability

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set Python path (required for all operations)
export PYTHONPATH=/path/to/GEDI-ORDER
```

### Training
```bash
# Basic training
python main/train.py --datadir /data/path \
    --pos_dir /positive/samples \
    --neg_dir /negative/samples \
    --which_model resnet50 \
    --epochs 20 \
    --batch_size 32 \
    --use_wandb 1

# Using shell script
./train.sh
```

### Model Deployment
```bash
# Deploy trained model
python deploy/deploy.py --parent /results/dir \
    --model_path /path/to/model.keras \
    --im_dir /images/to/classify \
    --preprocess_tfrecs 1 \
    --use_gedi_cnn 0

# Using shell script
./deploy.sh
```

### Grad-CAM Visualization
```bash
# Generate activation maps
python activationmap/gradcam.py --im_dir /images \
    --model_path /path/to/model.keras \
    --layer_name conv5_block3_out \
    --resdir /output/dir

# Using shell script
./gradcam.sh
```

## Configuration System

The `param_gedi.py` file contains machine-specific paths and training hyperparameters. Key configuration areas:

- **Machine paths**: Automatically detects hostname and sets appropriate data/model paths
- **Model parameters**: Architecture choice, learning rates, batch sizes
- **Data pipeline**: Image dimensions, augmentation settings, class weights
- **TFRecord paths**: Training, validation, and test dataset locations

When adding new machines, update the hostname dictionaries in `param_gedi.py:79-105`.

## Data Pipeline

The system uses TensorFlow's TFRecord format for efficient data handling:

1. **Image preprocessing**: Convert images → TFRecords via `preprocessing/create_tfrecs_from_lst.py`
2. **Data generation**: `preprocessing/datagenerator.py` handles batching, augmentation, and normalization
3. **Deployment**: `preprocessing/create_tfrecs_deploy.py` for inference data preparation

Input images are expected as single-channel (grayscale) at 200x200 pixels, then resized to 224x224x3 for CNN compatibility.

## Model Architecture Support

The system supports multiple CNN architectures via the `which_model` parameter:
- **VGG16/VGG19**: Original GEDI-CNN architecture
- **ResNet50**: Current recommended architecture with better gradient flow

Models are trained with transfer learning, using ImageNet pretrained weights and fine-tuning for cell classification.

## Weights and Biases Integration

Training supports Weights & Biases logging when `use_wandb=1`. Required environment variables:
- `WANDB_ENTITY`: W&B team/user
- `WANDB_PROJECT`: Project name
- `WANDB_RUN_GROUP`: Experiment grouping
- `WANDB_NAME`: Specific run name

## Testing

No formal test suite is present. Validation occurs through:
- Model performance metrics during training
- Deploy script accuracy evaluation
- Grad-CAM visual inspection for biological relevance

## Important Notes

- **PYTHONPATH**: Must be set to project root for all operations
- **GPU/CPU**: Deploy script forces CPU execution to avoid CuDNN conflicts
- **File paths**: All scripts use absolute paths; update `param_gedi.py` for new environments
- **TFRecord naming**: Deploy creates files with label suffixes (`deploy_0.tfrecord`, `deploy_1.tfrecord`)