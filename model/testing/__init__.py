"""
FLAME AI Model Testing Suite

This package contains testing tools for evaluating existing trained
FLAME AI wildfire prediction models.

Available modules:
- compare_existing_models: Compare pre-trained model files on test data
- visualize_results: Model prediction visualization and analysis
"""

__version__ = "1.0.0"
__author__ = "FLAME AI Team"

# Import main classes for existing model analysis
from .compare_existing_models import ExistingModelComparator
from .visualize_results import ModelVisualizer

__all__ = [
    'ExistingModelComparator',
    'ModelVisualizer'
] 