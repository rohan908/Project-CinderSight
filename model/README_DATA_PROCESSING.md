# Enhanced NDWS Dataset Processing Tools (PyTorch)

This directory contains PyTorch-based tools for processing the **Enhanced Next Day Wildfire Spread (NDWS) dataset** with 19 input features, as described in the paper by Rufai Yusuf Zakari et al.

## Overview

The dataset enhancement includes:

*   **19 input features** (vs original 12): weather, terrain, vegetation, human factors
*   **Weather forecasts**: Next-day temperature, precipitation, wind speed/direction
*   **Terrain features**: Slope, aspect (in addition to elevation)
*   **Extended temporal coverage**: 2015-2024 (vs 2012-2020)
*   **PyTorch compatibility**: Native tensor operations and efficient data loading

## Files

*   `run_data_processing.py` - **Main runner script with centralized path validation**
*   `data_visualizer.py` - Visualize dataset features and statistics
*   `enhanced_data_cleaner.py` - Clean and process TFRecord data to PyTorch tensors
*   `README_DATA_PROCESSING.md` - This documentation

## Quick Start

### Prerequisites

```
pip install torch numpy matplotlib seaborn pathlib tfrecord
```

### Basic Usage

```
cd model

# 1. Check if everything is set up correctly
python run_data_processing.py --check

# 2. Run data visualization (creates charts and statistics)
python run_data_processing.py --visualize

# 3. Clean and process the data (converts TFRecord → PyTorch tensors)
python run_data_processing.py --clean

# 4. Or run both together
python run_data_processing.py --visualize --clean
```

### Custom Paths

```
# Specify custom directories
python run_data_processing.py --visualize --clean \
  --data-dir "path/to/tfrecords" \
  --vis-output "path/to/visualizations" \
  --clean-output "path/to/processed"
```

## Path Validation & Safety Features

The runner now includes **robust path validation**:

*   **Input validation**: Checks data directory exists and contains TFRecord files
*   **Output directory creation**: Automatically creates output directories
*   **Permission checks**: Validates write permissions before processing
*   **File pattern validation**: Ensures train/test/eval TFRecord files are present
*   **Detailed error messages**: Clear guidance when issues are found

### Expected Directory Structure

```
model/data/raw/                    # Input TFRecord files
├── enhanced_train_*.tfrecord      # Training data
├── enhanced_test_*.tfrecord       # Test data  
└── enhanced_eval_*.tfrecord       # Evaluation data

model/visualizations/              # Generated visualizations
├── feature_distributions.png
├── fire_progression.png
├── feature_correlations.png
└── multimodal_sample.png

model/data/processed/              # Processed PyTorch data
├── train.data                     # Training features (N, 19, 64, 64)
├── train.labels                   # Training labels (N, 64, 64)
├── test.data                      # Test features  
├── test.labels                    # Test labels
├── validation.data                # Validation features
├── validation.labels              # Validation labels
└── metadata.pkl                   # Dataset metadata
```

## Enhanced Features (19 Total)

### Weather Factors (Current Day) - 8 features

*   `vs` - Wind speed (m/s)
*   `pr` - Precipitation (mm)
*   `sph` - Specific humidity
*   `tmmx` - Max temperature (K)
*   `tmmn` - Min temperature (K)
*   `th` - Wind direction (degrees)
*   `erc` - Energy release component (BTU/sq ft)
*   `pdsi` - Palmer Drought Severity Index

### Weather Forecasts (Next Day) - 4 features

*   `ftemp` - Forecast temperature (K)
*   `fpr` - Forecast precipitation (mm)
*   `fws` - Forecast wind speed (m/s)
*   `fwd` - Forecast wind direction (degrees)

### Terrain Factors - 3 features

*   `elevation` - Elevation (m)
*   `aspect` - Aspect (degrees)
*   `slope` - Slope (degrees)

### Vegetation - 2 features

*   `NDVI` - Normalized Difference Vegetation Index
*   `EVI` - Enhanced Vegetation Index

### Human Factors - 1 feature

*   `population` - Population density (people/sq km)

### Fire Context - 1 feature

*   `PrevFireMask` - Previous fire mask (-1: unlabeled, 0: no fire, 1: fire)

## Data Processing Details

### Normalization Strategy

*   **Weather/terrain/vegetation**: Clipped to percentiles, then z-score normalized
*   **Fire masks**: No normalization (categorical: -1, 0, 1)

### Data Quality Control

*   **Missing data removal**: Samples with missing fire mask data (-1 values) are filtered
*   **Spatial resolution**: Maintains 64×64 resolution (1km per pixel)
*   **Channel ordering**: Features → (N, 19, H, W), Labels → (N, H, W)

## Usage with FLAME AI Model (PyTorch)

After processing, the data is ready for the FLAME AI + Enhanced NDWS model:

```python
import pickle
import torch
import numpy as np

# Load processed data
with open('model/data/processed/train.data', 'rb') as f:
    train_features = pickle.load(f)  # Shape: (N, 19, 64, 64)

with open('model/data/processed/train.labels', 'rb') as f:
    train_labels = pickle.load(f)   # Shape: (N, 64, 64)

# Convert to PyTorch tensors
train_features = torch.from_numpy(train_features).float()
train_labels = torch.from_numpy(train_labels).float()

# Load metadata
with open('model/data/processed/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"Features: {metadata['input_features']}")
print(f"Training samples: {len(train_features)}")
print(f"Framework: {metadata['framework']}")
```

## Visualizations Generated

1.  **Feature Distributions** - Histograms of all 19 input features
2.  **Fire Progression** - Before/after fire masks showing spread patterns
3.  **Feature Correlations** - Correlation matrix between all features
4.  **Multimodal Sample** - All features for a single sample with fire masks

## Error Handling & Troubleshooting

### Path validation errors

The runner will clearly indicate path issues:

```
ERROR: Input data directory does not exist: /path/to/data
Please ensure the data directory exists and contains TFRecord files.
```

### Missing TFRecord files

```
ERROR: Missing required TFRecord files in model/data/raw:
Missing patterns: *train*.tfrecord, *eval*.tfrecord
Please ensure you have training, test, and evaluation TFRecord files.
```

### Import errors

*   Ensure all required packages are installed: `torch`, `numpy`, `matplotlib`, `seaborn`, `tfrecord`
*   Run `python run_data_processing.py --check`

### Memory issues

*   The processing loads entire datasets into memory
*   For large datasets, ensure sufficient RAM (8GB+ recommended)
*   Consider processing subsets if memory is limited

### TFRecord format issues

*   Ensure TFRecord files contain the expected 19 features + FireMask
*   Check that features are stored as 64×64 float arrays
*   Verify file integrity if parsing errors occur

## Command Reference

```
# Check requirements and paths
python run_data_processing.py --check

# Basic operations
python run_data_processing.py --visualize
python run_data_processing.py --clean
python run_data_processing.py --visualize --clean

# Custom paths
python run_data_processing.py --visualize --clean \
  --data-dir "custom/data/path" \
  --vis-output "custom/vis/output" \
  --clean-output "custom/processed/output"
```

### Arguments

*   `--check` - Validate requirements and paths
*   `--visualize` - Generate dataset visualizations
*   `--clean` - Process TFRecord data to PyTorch format
*   `--data-dir` - Input directory with TFRecord files (default: `model/data/raw`)
*   `--vis-output` - Visualization output directory (default: `model/visualizations`)
*   `--clean-output` - Processed data output directory (default: `model/data/processed`)

## Success Indicators

When everything works correctly, you'll see:

```
SUCCESS: All paths validated successfully!
SUCCESS: Data visualization completed successfully!
SUCCESS: Data cleaning and processing completed successfully!
SUCCESS: All operations completed successfully!
   Visualizations saved to: model/visualizations
   Processed data saved to: model/data/processed
```