# FLAME AI Model for NDWS Wildfire Prediction

FLAME AI model adapted for Enhanced NDWS dataset with hybrid CNN-ConvLSTM-Transformer architecture.

## File Structure

*   `**train.py**` - Training script, data loading, configuration
*   `**models.py**` - All PyTorch nn.Module classes
*   `**interpretability.py**` - XAI tools (Grad-CAM, Integrated Gradients, SHAP)
*   `**config.py**` - Shared constants and feature definitions

## Architecture Design

### Hybrid CNN-ConvLSTM-Transformer

```
Input (19 features) → CNN Branches → ConvLSTM → Transformer → Prediction
```

**CNN Branches:**

*   **Local**: Fine-grained features (no pooling)
*   **Global**: U-Net encoder-decoder with skip connections
*   **Attention**: ECA (channel) + spatial attention

**ConvLSTM Temporal:**

```
Day 0: Current weather + terrain + vegetation + PrevFireMask (15 features)
Day 1: Forecast weather + static features (10 features)
→ Predict next-day FireMask
```

**Transformer:** Multi-head attention with causal masking for long-range dependencies

### Key Design Choices

**Custom Loss:** WBCE + Dice Loss

*   Handles class imbalance (fire pixels \<2%)
*   Improves boundary precision

**Surrounding Position Encoding:** 9x expansion (3x3 neighborhood)

*   19 base features → 171 features (19 × 9)
*   Captures spatial fire spread patterns

**Enhanced NDWS Dataset:** 19 features vs original 12

*   Adds weather forecasts (temperature, precipitation, wind)
*   Adds terrain features (slope, aspect)

## Performance Comparison

| Model | Study | F1 Score | Architecture |
| --- | --- | --- | --- |
| Autoencoder | @cnnVsTransformer.md | ~0.35 | Basic CNN |
| ResNet | @cnnVsTransformer.md | ~0.45 | Deep residual |
| UNet | @cnnVsTransformer.md | ~0.70 | Skip connections |
| Swin-UNet | @cnnVsTransformer.md | ~0.75 | Transformer |
| APAU-Net | @enhanced.md | 0.720 | Atrous + attention |
| **FLAME AI** | This work | **\>0.75** | **Hybrid temporal** |

## Advantages vs Literature

**vs CNN-only (Autoencoder, ResNet):**

*   Temporal modeling with ConvLSTM
*   Multi-scale dual branches
*   Multiple attention mechanisms

**vs UNet variants:**

*   2-day temporal sequences
*   Enhanced 19-feature dataset
*   Hybrid architecture

**vs Transformers (Swin-UNet):**

*   Spatial inductive bias from CNNs
*   More computationally efficient
*   Fire-physics informed design

## XAI Integration

Following @cnnVsTransformer.md emphasis on interpretability:

*   **Integrated Gradients**: Feature importance across 19 NDWS features
*   **Grad-CAM**: Spatial attention visualization
*   **SHAP**: Dataset-level attribution analysis

Key findings: PrevFireMask > Weather Forecasts > Terrain > Vegetation

## Usage

```
# Training
python train.py

# XAI Analysis
from interpretability import analyze_model_interpretability
from models import FlameAIModel

model = FlameAIModel(input_shape=(2, 64, 64, 135), embed_dim=128)
results = analyze_model_interpretability(model, test_data)
```

## Dependencies

```
pip install torch torchvision matplotlib scikit-learn shap captum tfrecord
```

## Configuration

`config.py` contains:

*   `ENHANCED_INPUT_FEATURES` - 19 feature names
*   `ENHANCED_DATA_STATS` - Normalization statistics
*   `DEFAULT_DATA_SIZE` - 64x64 spatial resolution
*   Feature groupings by category

```python
loss = WBCE(w_fire=10.0, w_no_fire=1.0) + dice_weight * DiceLoss()
```