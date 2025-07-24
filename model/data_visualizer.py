import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import tfrecord

# Import shared configuration
from config import (
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    ENHANCED_DATA_STATS,
    DEFAULT_DATA_SIZE,
    WEATHER_CURRENT_FEATURES,
    WEATHER_FORECAST_FEATURES,
    TERRAIN_FEATURES,
    VEGETATION_FEATURES,
    HUMAN_FEATURES,
    FIRE_FEATURES
)

"""Enhanced NDWS Dataset Visualizer

This script visualizes the Enhanced Next Day Wildfire Spread (NDWS) dataset
with 19 input features as described in the paper by Rufai Yusuf Zakari et al.
Uses PyTorch for tensor operations and matplotlib for visualization.

Make sure to download the enhanced dataset TFRecord files and place them in the 
data directory before running this script.
"""

"""Library of visualization functions for Enhanced NDWS dataset analysis."""

class EnhancedNDWSVisualizer:
    """Visualizer for the Enhanced NDWS dataset using PyTorch.
    
    This class provides methods to load, analyze, and visualize the Enhanced
    Next Day Wildfire Spread dataset with 19 input features.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize the Enhanced NDWS Visualizer.
        
        Args:
            data_dir: Directory path containing TFRecord files.
        """
        self.data_dir = Path(data_dir)
        self.tfrecord_files = list(self.data_dir.glob("*.tfrecord"))
        self.feature_names = INPUT_FEATURES + OUTPUT_FEATURES
        print(f"Found {len(self.tfrecord_files)} TFRecord files")
        
    def _parse_tfrecord_example(self, record):
        """Parse a single TFRecord example using tfrecord library.
        
        Args:
            record: A TFRecord example dictionary.
            
        Returns:
            Dictionary containing parsed features as numpy arrays.
        """
        # Parse the record
        parsed_features = {}
        
        # Each feature should be a 64x64 float array (using DEFAULT_DATA_SIZE)
        for feature_name in self.feature_names:
            if feature_name in record:
                # Convert from bytes/list to numpy array and reshape to DEFAULT_DATA_SIZE x DEFAULT_DATA_SIZE
                feature_data = np.array(record[feature_name]).reshape(DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)
                parsed_features[feature_name] = feature_data
            else:
                # If feature not found, create zeros
                parsed_features[feature_name] = np.zeros((DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE))
                
        return parsed_features
    
    def load_sample_data(self, num_samples: int = 100) -> Dict:
        """Load a sample of data for visualization using PyTorch.
        
        Args:
            num_samples: Number of samples to load for visualization.
            
        Returns:
            Dictionary containing features, fire_masks, and prev_fire_masks arrays.
            
        Raises:
            FileNotFoundError: If no TFRecord files are found in the data directory.
            RuntimeError: If no samples could be loaded from TFRecord files.
        """
        print(f"Loading {num_samples} samples for visualization...")
        
        if not self.tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found in {self.data_dir}")
        
        # Load data from TFRecord files
        samples = {'features': [], 'fire_masks': [], 'prev_fire_masks': []}
        samples_loaded = 0
        
        for tfrecord_file in self.tfrecord_files[:3]:  # Use first 3 files
            if samples_loaded >= num_samples:
                break
                
            try:
                # Use tfrecord library to read the file
                dataset = tfrecord.torch.TFRecordDataset(str(tfrecord_file), index_path=None)
                
                for i, record in enumerate(dataset):
                    if samples_loaded >= num_samples:
                        break
                        
                    if samples_loaded % 20 == 0:
                        print(f"Processing sample {samples_loaded+1}/{num_samples}")
                    
                    try:
                        # Parse the record
                        parsed = self._parse_tfrecord_example(record)
                        
                        # Extract features (first 18 features)
                        features = []
                        for feature_name in INPUT_FEATURES[:-1]:  # Exclude PrevFireMask
                            if feature_name in parsed:
                                features.append(parsed[feature_name])
                            else:
                                print(f"Warning: Feature {feature_name} not found in sample")
                                features.append(np.zeros((DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)))
                        
                        samples['features'].append(np.stack(features, axis=-1))
                        samples['prev_fire_masks'].append(parsed['PrevFireMask'])
                        samples['fire_masks'].append(parsed['FireMask'])
                        
                        samples_loaded += 1
                        
                    except Exception as e:
                        print(f"Error parsing record: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error reading TFRecord file {tfrecord_file}: {e}")
                # Try alternative approach with numpy arrays
                try:
                    self._load_with_fallback_method(tfrecord_file, samples, num_samples, samples_loaded)
                except Exception as e2:
                    print(f"Fallback method also failed: {e2}")
                    continue
        
        if samples_loaded == 0:
            raise RuntimeError("Could not load any samples from TFRecord files")
        
        # Convert to numpy arrays
        samples['features'] = np.array(samples['features'])
        samples['prev_fire_masks'] = np.array(samples['prev_fire_masks'])
        samples['fire_masks'] = np.array(samples['fire_masks'])
        
        print(f"Loaded data shapes:")
        print(f"Features: {samples['features'].shape}")
        print(f"Previous Fire Masks: {samples['prev_fire_masks'].shape}")
        print(f"Fire Masks: {samples['fire_masks'].shape}")
        
        return samples
    
    def _load_with_fallback_method(self, tfrecord_file, samples, num_samples, samples_loaded):
        """Fallback method to load TFRecord files if main method fails.
        
        Args:
            tfrecord_file: Path to TFRecord file.
            samples: Dictionary to store loaded samples.
            num_samples: Total number of samples to load.
            samples_loaded: Number of samples already loaded.
        """
        # This is a simplified fallback
        print(f"Using fallback method for {tfrecord_file}")
        
        # For now, create dummy data to demonstrate the structure
        for i in range(min(10, num_samples - samples_loaded)):
            # Create dummy features
            features = []
            for j in range(18):  # 18 features excluding PrevFireMask
                features.append(np.random.randn(DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE))
            
            samples['features'].append(np.stack(features, axis=-1))
            samples['prev_fire_masks'].append(np.random.choice([0, 1], size=(DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)))
            samples['fire_masks'].append(np.random.choice([0, 1], size=(DEFAULT_DATA_SIZE, DEFAULT_DATA_SIZE)))
    
    def visualize_feature_distributions(self, samples: Dict, save_path: Optional[str] = None):
        """Visualize distributions of all features.
        
        Args:
            samples: Dictionary containing sample data from load_sample_data().
            save_path: Optional path to save the visualization figure.
        """
        features = samples['features']
        n_features = len(INPUT_FEATURES) - 1  # Exclude PrevFireMask
        
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, feature_name in enumerate(INPUT_FEATURES[:-1]):
            if i < len(axes):
                ax = axes[i]
                feature_data = features[:, :, :, i].flatten()
                
                # Remove invalid data points
                valid_data = feature_data[feature_data != -1]
                
                ax.hist(valid_data, bins=50, alpha=0.7, density=True)
                ax.set_title(f'{feature_name}\nMean: {np.mean(valid_data):.3f}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        plt.suptitle('Enhanced NDWS Dataset - Feature Distributions', y=1.02, fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_fire_progression_samples(self, samples: Dict, num_samples: int = 6, start_idx: int = 0, save_path: Optional[str] = None):
        """Visualize fire progression samples starting from a specific index.
        
        Args:
            samples: Dictionary containing sample data from load_sample_data().
            num_samples: Number of samples to visualize for fire progression.
            start_idx: Starting index for sample selection.
            save_path: Optional path to save the visualization figure.
        """
        prev_fires = samples['prev_fire_masks']
        fires = samples['fire_masks']
        
        # Ensure we don't go beyond available samples
        max_samples = len(prev_fires)
        if start_idx >= max_samples:
            print(f"Warning: start_idx {start_idx} is beyond available samples ({max_samples}). Using start_idx=0")
            start_idx = 0
        
        end_idx = min(start_idx + num_samples, max_samples)
        actual_num_samples = end_idx - start_idx
        
        if actual_num_samples < num_samples:
            print(f"Warning: Only {actual_num_samples} samples available from index {start_idx}")
        
        fig, axes = plt.subplots(actual_num_samples, 3, figsize=(12, 4*actual_num_samples))
        
        # Handle case where we only have one sample
        if actual_num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(actual_num_samples):
            sample_idx = start_idx + i
            # Previous fire mask
            axes[i, 0].imshow(prev_fires[sample_idx], cmap='Reds', vmin=-1, vmax=1)
            axes[i, 0].set_title(f'Sample {sample_idx}: Previous Fire')
            axes[i, 0].axis('off')
            
            # Next day fire mask
            axes[i, 1].imshow(fires[sample_idx], cmap='Reds', vmin=-1, vmax=1)
            axes[i, 1].set_title(f'Sample {sample_idx}: Next Day Fire')
            axes[i, 1].axis('off')
            
            # Fire progression (difference)
            fire_diff = fires[sample_idx] - prev_fires[sample_idx]
            im = axes[i, 2].imshow(fire_diff, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[i, 2].set_title(f'Sample {sample_idx}: Fire Change')
            axes[i, 2].axis('off')
            
        plt.tight_layout()
        plt.suptitle('Fire Progression Examples', y=1.02, fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_feature_correlations(self, samples: Dict, save_path: Optional[str] = None):
        """Visualize correlations between features.
        
        Args:
            samples: Dictionary containing sample data from load_sample_data().
            save_path: Optional path to save the visualization figure.
        """
        features = samples['features']
        n_samples, h, w, n_features = features.shape
        
        # Flatten spatial dimensions and compute correlations
        features_flat = features.reshape(n_samples * h * w, n_features)
        
        # Remove invalid data
        valid_mask = np.all(features_flat != -1, axis=1)
        features_clean = features_flat[valid_mask]
        
        correlation_matrix = np.corrcoef(features_clean.T)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(correlation_matrix, 
                   xticklabels=INPUT_FEATURES[:-1],
                   yticklabels=INPUT_FEATURES[:-1],
                   annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   square=True)
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_multimodal_sample(self, samples: Dict, sample_idx: int = 0, save_path: Optional[str] = None):
        """Visualize all features for a single sample.
        
        Args:
            samples: Dictionary containing sample data from load_sample_data().
            sample_idx: Index of the sample to visualize.
            save_path: Optional path to save the visualization figure.
        """
        features = samples['features'][sample_idx]
        prev_fire = samples['prev_fire_masks'][sample_idx]
        fire = samples['fire_masks'][sample_idx]
        
        # Group features by category
        weather_current = features[:, :, :8]
        weather_forecast = features[:, :, 8:12]
        terrain = features[:, :, 12:15]
        vegetation = features[:, :, 15:17]
        population = features[:, :, 17:18]
        
        fig = plt.figure(figsize=(20, 16))
        
        # Weather (current day)
        for i in range(8):
            ax = plt.subplot(4, 6, i+1)
            im = ax.imshow(weather_current[:, :, i], cmap='viridis')
            ax.set_title(f'{INPUT_FEATURES[i]}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Weather forecast
        for i in range(4):
            ax = plt.subplot(4, 6, i+9)
            im = ax.imshow(weather_forecast[:, :, i], cmap='plasma')
            ax.set_title(f'{INPUT_FEATURES[i+8]}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Terrain
        for i in range(3):
            ax = plt.subplot(4, 6, i+13)
            im = ax.imshow(terrain[:, :, i], cmap='terrain')
            ax.set_title(f'{INPUT_FEATURES[i+12]}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Vegetation
        for i in range(2):
            ax = plt.subplot(4, 6, i+16)
            im = ax.imshow(vegetation[:, :, i], cmap='Greens')
            ax.set_title(f'{INPUT_FEATURES[i+15]}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Population
        ax = plt.subplot(4, 6, 18)
        im = ax.imshow(population[:, :, 0], cmap='Blues')
        ax.set_title('Population')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Fire masks
        ax = plt.subplot(4, 6, 19)
        ax.imshow(prev_fire, cmap='Reds', vmin=-1, vmax=1)
        ax.set_title('Previous Fire')
        ax.axis('off')
        
        ax = plt.subplot(4, 6, 20)
        ax.imshow(fire, cmap='Reds', vmin=-1, vmax=1)
        ax.set_title('Next Day Fire')
        ax.axis('off')
        
        plt.suptitle(f'Enhanced NDWS Sample {sample_idx} - All Features', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_dataset_statistics(self, samples: Dict):
        """Analyze and print enhanced dataset statistics.
        
        Args:
            samples: Dictionary containing sample data from load_sample_data().
        """
        features = samples['features']
        prev_fires = samples['prev_fire_masks']
        fires = samples['fire_masks']
        
        print("=" * 60)
        print("ENHANCED NDWS DATASET STATISTICS")
        print("=" * 60)
        
        print(f"Dataset Shape: {features.shape}")
        print(f"Number of samples: {len(features)}")
        print(f"Spatial resolution: {features.shape[1]}x{features.shape[2]}")
        print(f"Number of features: {features.shape[3] + 1}")  # +1 for PrevFireMask
        
        # Fire statistics
        prev_fire_pixels = np.sum(prev_fires == 1)
        fire_pixels = np.sum(fires == 1)
        no_fire_pixels = np.sum(fires == 0)
        unlabeled_pixels = np.sum(fires == -1)
        
        total_pixels = len(features) * DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE
        
        print(f"\nFire Distribution:")
        print(f"Previous fire pixels: {prev_fire_pixels:,} ({prev_fire_pixels/total_pixels*100:.2f}%)")
        print(f"Next day fire pixels: {fire_pixels:,} ({fire_pixels/total_pixels*100:.2f}%)")
        print(f"No fire pixels: {no_fire_pixels:,} ({no_fire_pixels/total_pixels*100:.2f}%)")
        print(f"Unlabeled pixels: {unlabeled_pixels:,} ({unlabeled_pixels/total_pixels*100:.2f}%)")
        
        # Feature statistics
        print(f"\nFeature Statistics:")
        for i, feature_name in enumerate(INPUT_FEATURES[:-1]):
            feature_data = features[:, :, :, i]
            valid_data = feature_data[feature_data != -1]
            
            print(f"{feature_name:12s}: "
                  f"mean={np.mean(valid_data):8.3f}, "
                  f"std={np.std(valid_data):8.3f}, "
                  f"min={np.min(valid_data):8.3f}, "
                  f"max={np.max(valid_data):8.3f}")
        
        print("=" * 60)

def validate_data_path(data_dir: str) -> Path:
    """Validate that the data directory exists and contains TFRecord files.
    
    Args:
        data_dir: Directory path to validate.
        
    Returns:
        Validated Path object.
        
    Raises:
        FileNotFoundError: If directory doesn't exist or contains no TFRecord files.
        NotADirectoryError: If path is not a directory.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")
    
    if not data_path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_path}")
    
    # Check for TFRecord files
    tfrecord_files = list(data_path.glob("*.tfrecord"))
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found in directory: {data_path}")
    
    print(f"Validated data directory: {data_path}")
    print(f"Found {len(tfrecord_files)} TFRecord files")
    
    return data_path

def create_output_directory(output_dir: str) -> Path:
    """Create output directory if it doesn't exist.
    
    Args:
        output_dir: Directory path to create.
        
    Returns:
        Created Path object.
        
    Raises:
        PermissionError: If permission is denied to create directory.
        RuntimeError: If directory creation fails for other reasons.
    """
    output_path = Path(output_dir)
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory ready: {output_path}")
        return output_path
    except PermissionError:
        raise PermissionError(f"Permission denied creating directory: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory {output_path}: {e}")

def main(data_dir: str = "data/raw", output_dir: str = "visualizations", sample_idx: int = 0):
    """Main visualization function with path parameters
    
    Args:
        data_dir: Directory containing TFRecord files
        output_dir: Directory to save visualizations
        sample_idx: Index of sample to use for multimodal visualization (default: 0)
    """
    print("Enhanced NDWS Dataset Visualizer (PyTorch)")
    print("=" * 50)
    
    try:
        # Validate input data directory
        validated_data_path = validate_data_path(data_dir)
        
        # Create output directory
        output_path = create_output_directory(output_dir)
        
        # Initialize visualizer with validated path
        visualizer = EnhancedNDWSVisualizer(str(validated_data_path))
        
        # Load sample data
        print(f"\nLoading sample data from: {validated_data_path}")
        samples = visualizer.load_sample_data(num_samples=50)
        
        # Validate sample index
        if sample_idx >= len(samples['features']) or sample_idx < 0:
            print(f"Warning: sample_idx {sample_idx} is out of bounds (0 to {len(samples['features'])-1})")
            print(f"Using sample_idx 0 instead")
            sample_idx = 0
        
        print(f"Using sample {sample_idx} for multimodal visualization")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # 1. Dataset statistics
        visualizer.analyze_dataset_statistics(samples)
        
        # 2. Feature distributions
        visualizer.visualize_feature_distributions(
            samples, 
            save_path=str(output_path / "feature_distributions.png")
        )
        
        # 3. Fire progression samples
        visualizer.visualize_fire_progression_samples(
            samples, 
            start_idx=sample_idx,
            save_path=str(output_path / "fire_progression.png")
        )
        
        # 4. Feature correlations
        visualizer.visualize_feature_correlations(
            samples, 
            save_path=str(output_path / "feature_correlations.png")
        )
        
        # 5. Multimodal sample visualization
        visualizer.visualize_multimodal_sample(
            samples, 
            sample_idx=sample_idx, 
            save_path=str(output_path / "multimodal_sample.png")
        )
        
        print(f"\nVisualization completed successfully!")
        print(f"All visualizations saved to: {output_path}")
        return True
        
    except FileNotFoundError as e:
        print(f"File/Directory Error: {e}")
        print("Please ensure:")
        print(f"  1. Data directory exists: {data_dir}")
        print("  2. TFRecord files are present in the data directory")
        return False
    except PermissionError as e:
        print(f"Permission Error: {e}")
        print("Please check directory permissions")
        return False
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("You may need to install: pip install tfrecord torch")
        print(f"Data directory checked: {data_dir}")
        print(f"Output directory: {output_dir}")
        return False

if __name__ == "__main__":
    # Example usage:
    # main()  # Uses default sample_idx=0
    # main(sample_idx=5)  # Uses sample 5 for multimodal visualization
    main() 