# FLAME AI Model Testing Suite

Simple tools for evaluating your existing trained FLAME AI wildfire prediction models.

## **Compare Your Existing Models**

If you have trained `.pth` model files and want to compare them on test data:

```bash
# Navigate to testing directory
cd Project-CinderSight/model/testing

# Compare two existing models
python compare_existing_models.py path/to/old_model.pth path/to/new_model.pth

# Example:
python compare_existing_models.py ../models/baseline_model.pth ../models/augmented_model.pth
```

**What it does**:
- Loads both trained models
- Evaluates them on your test data from `data/processed`
- Compares F1, IoU, precision, recall
- Measures inference speed
- Generates comparison charts
- Saves results to JSON

**Output files**:
- `existing_models_comparison.json` - Detailed metrics
- `existing_models_comparison.png` - Performance charts

## Available Tools

### 1. `compare_existing_models.py` - Compare Pre-trained Models

**Purpose**: Compare two already-trained model files on test data

**Features**:
- Load any `.pth` model files
- Automatic input size detection from model config
- Comprehensive performance comparison
- Inference speed benchmarking
- Visual comparison charts

**Usage**:
```bash
python compare_existing_models.py model1.pth model2.pth
```

**Perfect for**: When you have trained models and want to see which performs better

### 2. `visualize_results.py` - Model Prediction Visualization

**Purpose**: Visualize trained model predictions and interpretability

**Features**:
- Load and visualize saved models
- Generate prediction heatmaps
- Feature importance analysis
- Model interpretability with Grad-CAM

**Usage**:
```python
from visualize_results import ModelVisualizer

visualizer = ModelVisualizer('your_model.pth')
visualizer.visualize_sample_predictions(num_samples=5)
visualizer.visualize_feature_importance()
```

## Output Files

- `existing_models_comparison.json` - Pre-trained model comparison results
- `existing_models_comparison.png` - Pre-trained model comparison chart
- Various visualization PNGs from the visualizer

## Dependencies

The tools automatically import required modules from the parent directory:
- `train.py` - Data loading functions
- `models.py` - FLAME AI model architecture
- `interpretability.py` - Model analysis tools
- `config.py` - Dataset configuration constants

## Troubleshooting

**Import Errors**: Ensure you're running from the testing directory and parent modules are accessible

**Memory Issues**: The tools automatically limit evaluation to reasonable sample sizes

**CUDA Errors**: Tools automatically fall back to CPU if GPU unavailable

**Data Loading**: Ensure processed NDWS data exists in `../data/processed/`

## Example Usage

### Compare Two Models:
```bash
# Compare your trained models
python compare_existing_models.py ../models/old_model.pth ../models/new_model.pth
```

### Visualize a Model:
```bash
python -c "
from visualize_results import ModelVisualizer
v = ModelVisualizer('../models/your_best_model.pth')
v.visualize_sample_predictions(num_samples=3)
v.visualize_feature_importance()
"
```

That's it! Simple tools for analyzing your existing trained models. 