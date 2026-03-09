# Train Files Comparison

This document compares the three training scripts in the main/ directory, analyzing their differences, utilities, and specific use cases.

## Files Overview

### 1. `train.py` - Standard Training Script
- **Purpose**: Single model training with comprehensive WandB integration
- **Data limit**: 10,000 images per class (hardcoded cutoff)
- **Key features**:
  - Full WandB integration with custom evaluation callbacks
  - Enhanced model artifact logging
  - Comprehensive test accuracy calculation
  - Standard train/validation/test split

### 2. `train_multiModel.py` - Multi-Timepoint Training
- **Purpose**: Trains separate models for each timepoint directory
- **Data organization**: Timepoint-based subdirectories (e.g., T0_0, T1_12, T12_144)
- **Key features**:
  - Creates separate models for each timepoint
  - Timepoint-specific directory organization
  - Individual WandB runs for each timepoint
  - Robust error handling for insufficient data

### 3. `train_multipath.py` - Multi-Path Combined Training
- **Purpose**: Combined training across all timepoints in a single model
- **Data organization**: Timepoint-based subdirectories combined into single dataset
- **Key features**:
  - Flattens all timepoint data into single training set
  - Custom subfolder evaluation callbacks
  - Per-subfolder accuracy tracking in WandB
  - Confusion matrix generation per subfolder

## Detailed Comparison

| Feature | train.py | train_multiModel.py | train_multipath.py |
|---------|----------|--------------------|--------------------|
| **Training Strategy** | Single model | Multiple models (one per timepoint) | Single model (all timepoints) |
| **Data Limit** | 10,000/class | No artificial limit | No artificial limit |
| **Directory Structure** | Flat (live/dead) | Nested (timepoint folders) | Nested (timepoint folders) |
| **WandB Integration** | Full with artifacts | Per-timepoint runs | Combined with subfolder metrics |
| **GPU Usage** | Auto-detect | Forced CPU | Forced CPU |
| **Model Output** | Single saved model | Multiple timepoint models | Single combined model |

## Key Technical Differences

### 1. Data Handling
- **train.py**: Uses `gather_imgs()` for flat directory structure
- **train_multiModel.py** & **train_multipath.py**: Use timepoint-aware `gather_imgs()` for nested structure

### 2. Training Loop
- **train.py**: Single training call with standard callbacks
- **train_multiModel.py**: Iterates through timepoints, trains separate models
- **train_multipath.py**: Single training with combined data from all timepoints

### 3. WandB Integration
- **train.py**:
  - Optional WandB with environment variable configuration
  - Model artifact logging
  - Custom evaluation callbacks for ground truth/predictions
- **train_multiModel.py**:
  - Per-timepoint WandB runs with `reinit=True`
  - Timepoint-specific project naming
- **train_multipath.py**:
  - Single WandB run with subfolder-specific metrics
  - Confusion matrices per subfolder

### 4. Model Architecture Support
All files support the same model architectures:
- VGG16, VGG19
- ResNet50
- Custom models (custom1, custom2)

### 5. File Format Differences
- **train.py**: Uses `.keras` format for model saving
- **train_multiModel.py**: Uses `.keras` format consistently
- **train_multipath.py**: Mixed - some `.h5`, some `.hdf5` in retrain mode

## Use Case Recommendations

### Use `train.py` when:
- Working with simple live/dead image classification
- Need comprehensive WandB tracking and model artifacts
- Working with images in flat directory structure
- Want single model for general classification

### Use `train_multiModel.py` when:
- Have time-series data with distinct timepoints
- Need separate models for each time condition
- Want to compare model performance across timepoints
- Have images organized by timepoint subdirectories

### Use `train_multipath.py` when:
- Want single model trained on all timepoint data
- Need detailed subfolder/timepoint performance analysis
- Want to understand how model performs across different conditions
- Have images organized by timepoint but prefer combined training

## Code Quality & Maintenance

### Similarities:
- All use same parameter system (`param_gedi.Param`)
- Identical model architectures and compilation
- Similar callback structures (early stopping, checkpoints, TensorBoard)
- Common evaluation methodology

### Notable Differences:
- **Error Handling**: `train_multiModel.py` has most robust error handling
- **Code Organization**: `train_multipath.py` has additional subfolder callback classes
- **Configuration**: `train.py` has most flexible WandB configuration
- **Documentation**: `train_multiModel.py` and `train_multipath.py` have better inline documentation

## Recommendations for Future Development

1. **Consolidate Common Code**: Extract shared functionality into base classes
2. **Standardize File Formats**: Use `.keras` consistently across all files
3. **Improve Error Handling**: Apply `train_multiModel.py`'s error handling to other files
4. **Configuration Management**: Standardize WandB configuration approach
5. **Documentation**: Add comprehensive docstrings to all methods