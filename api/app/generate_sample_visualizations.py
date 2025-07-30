import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import zoom

# Import config from the local config module
from .config import ENHANCED_INPUT_FEATURES, DEFAULT_DATA_SIZE, FEATURE_DESCRIPTIONS, FEATURE_CATEGORIES


# Import models and train functions from the local modules
try:
    from .models import FlameAIModel
    from .train import calculate_segmentation_metrics
except ImportError:
    # Create dummy classes if modules are not available
    class FlameAIModel:
        def __init__(self, *args, **kwargs):
            pass
    
    def calculate_segmentation_metrics(*args, **kwargs):
        return {}

class SampleVisualizationGenerator:
    """Generate comprehensive visualizations for individual test samples"""
    
    def __init__(self, model_path="model_nfp.pth", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = None
        self.model_config = {}
        self.load_model()
        
    def load_model(self):
        """Load the trained model from checkpoint"""
        print(f"Loading model from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model_config = checkpoint.get('config', {})
        
        print(f"Model config: {self.model_config}")
        
        # Initialize model with saved config - use the actual config from the trained model
        self.model = FlameAIModel(
            input_shape=(
                self.model_config.get('crop_size', 32),
                self.model_config.get('crop_size', 32),
                self.model_config.get('num_features', 19 * 9)  # 19 features * 9 total pixels
            ),
            embed_dim=self.model_config.get('embed_dim', 128),
            num_heads=8,
            attention_dropout=0.1,
            dropout=0.2
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Expected input shape: {self.model_config.get('crop_size', 32)}x{self.model_config.get('crop_size', 32)}x{self.model_config.get('num_features', 19 * 9)}")
        
    def load_test_sample(self, sample_idx: int, data_dir="data/processed"):
        """Load a specific test sample and preprocess it for the model"""
        print(f"Loading test sample {sample_idx}...")
        
        # This method is not used in Railway deployment
        # Data is loaded via Supabase and passed directly to generate_single_sample_with_data
        raise NotImplementedError("This method is not used in Railway deployment. Use generate_single_sample_with_data instead.")
    
    def _preprocess_for_model(self, features):
        """Preprocess features to match model expectations"""
        crop_size = self.model_config.get('crop_size', 32)
        expected_features = self.model_config.get('num_features', 171)
        
        # Handle different input shapes
        if features.shape[0] < features.shape[1]:  # (19, 64, 64) format
            # Transpose to (64, 64, 19) format
            features = features.transpose(1, 2, 0)
            print(f"  Transposed from (19, 64, 64) to (64, 64, 19)")
        
        # Crop to the expected size (center crop)
        h, w, c = features.shape
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        cropped = features[start_h:start_h+crop_size, start_w:start_w+crop_size, :]
        
        print(f"  Cropped from {h}x{w}x{c} to {crop_size}x{crop_size}x{c}")
        
        # Handle the feature dimension mismatch
        if expected_features > c:
            # The model expects more features than we have (171 vs 19)
            # This suggests the model was trained with temporal data
            # Create a workaround by replicating/padding features
            print(f"  Model expects {expected_features} features, we have {c}")
            print("  Creating feature expansion to match model expectations...")
            
            # Expand features by repeating them in a structured way
            expanded = np.zeros((crop_size, crop_size, expected_features))
            
            # Repeat the original features multiple times
            repeats = expected_features // c
            remainder = expected_features % c
            
            for i in range(repeats):
                expanded[:, :, i*c:(i+1)*c] = cropped
            if remainder > 0:
                # Fix: properly handle the remainder by taking the first 'remainder' features
                expanded[:, :, repeats*c:repeats*c+remainder] = cropped[:, :, :remainder]
                
            print(f"  Expanded to {crop_size}x{crop_size}x{expected_features}")
            return expanded
        else:
            return cropped[:, :, :expected_features]
    
    def _preprocess_target_for_model(self, target):
        """Preprocess target to match model expectations"""
        crop_size = self.model_config.get('crop_size', 32)
        
        # Handle different input shapes
        if len(target.shape) == 2:  # (64, 64) format
            # Add channel dimension
            target = target[:, :, np.newaxis]
            print(f"  Added channel dimension to target: {target.shape}")
        elif target.shape[0] < target.shape[1]:  # (1, 64, 64) format
            # Transpose to (64, 64, 1) format
            target = target.transpose(1, 2, 0)
            print(f"  Transposed target from (1, 64, 64) to (64, 64, 1)")
        
        # Crop to the expected size (center crop)
        h, w, c = target.shape
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        cropped = target[start_h:start_h+crop_size, start_w:start_w+crop_size, :]
        
        print(f"  Target cropped from {h}x{w}x{c} to {crop_size}x{crop_size}x{c}")
        return cropped
    
    def create_sample_directory(self, sample_idx: int, base_dir="visualizations"):
        """Create directory structure for sample visualizations"""
        sample_dir = Path(base_dir) / f"sample_{sample_idx}" / "visualizations"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Created visualization directory: {sample_dir}")
        return sample_dir
    
    def predict_sample(self, features):
        """Run inference on a single sample"""
        with torch.no_grad():
            # Add batch dimension and convert to tensor
            if not isinstance(features, torch.Tensor):
                features = torch.Tensor(features)
            features_batch = features.unsqueeze(0).to(self.device)
            prediction = self.model(features_batch)
            return prediction.cpu().numpy()[0]  # Remove batch dimension
    
    def generate_individual_feature_visualizations(self, features: np.ndarray, output_dir: Path):
        """Generate separate visualization for each input feature using original data"""
        print("Generating individual feature visualizations...")
        
        # Use original features for visualization - handle different formats
        viz_features = self.original_features
        
        # Ensure features are in (64, 64, 19) format for visualization
        if viz_features.shape[0] < viz_features.shape[1]:  # (19, 64, 64)
            viz_features = viz_features.transpose(1, 2, 0)  # -> (64, 64, 19)
            print(f"  Transposed features for visualization: {viz_features.shape}")
        
        generated_files = []
        
        for i, feature_name in enumerate(ENHANCED_INPUT_FEATURES):
            if i >= viz_features.shape[2]:
                break
                
            feature_data = viz_features[:, :, i]
            
            # Get feature description and unit
            feature_key = feature_name.lower().replace(' ', '').replace('(', '').replace(')', '')
            description = FEATURE_DESCRIPTIONS.get(feature_key, feature_name)
            
            # Determine category and colormap
            category_colormap = 'viridis'  # default
            for category_info in FEATURE_CATEGORIES.values():
                if feature_key in category_info['features']:
                    category_colormap = category_info['colormap']
                    break
            
            # Create clean filename
            clean_name = feature_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            filename = f"feature_{i:02d}_{clean_name}.png"
            filepath = output_dir / filename
            
            # Create visualization with proper labels
            plt.figure(figsize=(10, 8))
            im = plt.imshow(feature_data, cmap=category_colormap)
            
            # Enhanced title with description and unit
            plt.title(f'Feature {i}: {description}', fontsize=16, fontweight='bold')
            
            # Add colorbar with unit information
            cbar = plt.colorbar(im, shrink=0.8)
            if '(' in description and ')' in description:
                unit = description.split('(')[-1].split(')')[0]
                cbar.set_label(f'Value ({unit})', fontsize=12)
            else:
                cbar.set_label('Value', fontsize=12)
            
            # Add feature category information
            for category_name, category_info in FEATURE_CATEGORIES.items():
                if feature_key in category_info['features']:
                    plt.figtext(0.02, 0.02, f"Category: {category_info['description']}", 
                              fontsize=10, style='italic', color='gray')
                    break
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            generated_files.append(filename)
            print(f"  Generated: {filename} - {description}")
        
        return generated_files
    
    def generate_fire_progression_visualization(self, features: np.ndarray, target: np.ndarray, 
                                              prediction: np.ndarray, output_dir: Path):
        """Generate fire progression visualization with separate images"""
        print("Generating fire progression visualizations...")
        
        # Use original data for visualization (64x64)
        # Ensure features are in (64, 64, 19) format for visualization
        if self.original_features.shape[0] < self.original_features.shape[1]:  # (19, 64, 64)
            orig_features = self.original_features.transpose(1, 2, 0)  # -> (64, 64, 19)
            orig_prev_fire = orig_features[:, :, -1]  # Last feature is PrevFireMask
        else:  # (64, 64, 19)
            orig_prev_fire = self.original_features[:, :, -1]
            
        # Ensure target is in (64, 64) format for visualization
        if len(self.original_target.shape) == 2:  # (64, 64)
            orig_ground_truth = self.original_target
        else:  # (64, 64, 1) or (1, 64, 64)
            if self.original_target.shape[0] < self.original_target.shape[1]:  # (1, 64, 64)
                orig_ground_truth = self.original_target[0, :, :]
            else:  # (64, 64, 1)
                orig_ground_truth = self.original_target[:, :, 0]
        
        # Use prediction results (32x32) - resize for comparison
        pred_probability = prediction[:, :, 0]
        pred_binary = (pred_probability > 0.5).astype(float)
        
        # Resize prediction to match original dimensions for visualization
        orig_size = orig_ground_truth.shape[0]
        pred_size = pred_probability.shape[0]
        zoom_factor = orig_size / pred_size
        
        pred_probability_resized = zoom(pred_probability, zoom_factor, order=1)
        pred_binary_resized = zoom(pred_binary, zoom_factor, order=0)
        
        # Individual fire progression images with enhanced descriptions
        fire_images = {
            'previous_fire': (orig_prev_fire, 'Previous Day Fire Mask\n(Time t)', 'Reds'),
            'ground_truth': (orig_ground_truth, 'Ground Truth Fire Mask\n(Time t+1)', 'Reds'),
            'prediction_probability': (pred_probability_resized, 'Model Prediction\n(Probability)', 'Reds'),
            'prediction_binary': (pred_binary_resized, 'Model Prediction\n(Binary Threshold > 0.5)', 'Reds')
        }
        
        generated_files = []
        
        for name, (data, title, colormap) in fire_images.items():
            filename = f"fire_{name}.png"
            filepath = output_dir / filename
            
            plt.figure(figsize=(10, 8))
            plt.imshow(data, cmap=colormap, vmin=0, vmax=1)
            plt.title(title, fontsize=16, fontweight='bold')
            
            # Add colorbar with appropriate label
            cbar = plt.colorbar(shrink=0.8)
            if 'probability' in name:
                cbar.set_label('Probability (0-1)', fontsize=12)
            elif 'binary' in name:
                cbar.set_label('Fire Status (0=No Fire, 1=Fire)', fontsize=12)
            else:
                cbar.set_label('Fire Status (0=No Fire, 1=Fire)', fontsize=12)
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            generated_files.append(filename)
            print(f"  Generated: {filename}")
        
        # Comparison overlay using resized predictions
        comparison = np.zeros((orig_ground_truth.shape[0], orig_ground_truth.shape[1], 3))
        comparison[:, :, 0] = orig_ground_truth  # Red for ground truth
        comparison[:, :, 1] = pred_binary_resized   # Green for prediction
        comparison[:, :, 2] = 0
        
        filename = "fire_comparison_overlay.png"
        filepath = output_dir / filename
        
        plt.figure(figsize=(10, 8))
        plt.imshow(comparison)
        plt.title('Fire Spread Comparison\n(Red=Ground Truth, Green=Prediction, Yellow=Both)', fontsize=16, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Ground Truth Only'),
            Patch(facecolor='green', label='Prediction Only'),
            Patch(facecolor='yellow', label='Both Correct')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        generated_files.append(filename)
        print(f"  Generated: {filename}")
        
        return generated_files
    
    def generate_metrics_dashboard(self, prediction: np.ndarray, target: np.ndarray, output_dir: Path):
        """Generate separate metrics visualizations for the sample"""
        print("Generating metrics visualizations...")
        print(f"  Prediction shape: {prediction.shape}, Target shape: {target.shape}")
        print(f"  Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")
        print(f"  Target range: [{target.min():.3f}, {target.max():.3f}]")
        
        # Calculate metrics
        pred_tensor = torch.FloatTensor(prediction).unsqueeze(0)
        target_tensor = torch.FloatTensor(target).unsqueeze(0)
        
        # Additional debugging
        print(f"  Tensor shapes - pred: {pred_tensor.shape}, target: {target_tensor.shape}")
        print(f"  Tensor ranges - pred: [{pred_tensor.min():.3f}, {pred_tensor.max():.3f}], target: [{target_tensor.min():.3f}, {target_tensor.max():.3f}]")
        print(f"  Target unique values: {torch.unique(target_tensor)}")
        print(f"  Prediction unique values (after threshold): {torch.unique((pred_tensor > 0.5).float())}")
        
        metrics = calculate_segmentation_metrics(pred_tensor, target_tensor)
        
        # Debug the actual calculated values
        print(f"  Raw metrics from function:")
        print(f"    TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")
        print(f"    Precision: {metrics['precision']:.6f}")
        print(f"    Recall: {metrics['recall']:.6f}")
        print(f"    F1: {metrics['f1']:.6f}")
        print(f"    IoU: {metrics['iou']:.6f}")
        
        # Manual verification
        tp, fp, fn, tn = metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']
        manual_precision = tp / (tp + fp + 1e-6)
        manual_recall = tp / (tp + fn + 1e-6)
        manual_f1 = 2 * manual_precision * manual_recall / (manual_precision + manual_recall + 1e-6)
        manual_iou = tp / (tp + fp + fn + 1e-6)
        print(f"  Manual verification:")
        print(f"    Precision: {manual_precision:.6f}")
        print(f"    Recall: {manual_recall:.6f}")
        print(f"    F1: {manual_f1:.6f}")
        print(f"    IoU: {manual_iou:.6f}")
        
        generated_files = []
        
        # 1. Metrics bar chart
        plt.figure(figsize=(10, 6))
        metric_names = ['Precision', 'Recall', 'F1 Score', 'IoU']
        metric_values = [metrics['precision'], metrics['recall'], metrics['f1'], metrics['iou']]
        
        colors = ['skyblue', 'lightgreen', 'gold', 'lightcoral']
        bars = plt.bar(metric_names, metric_values, color=colors)
        plt.ylabel('Score', fontsize=12)
        plt.title('Sample Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        filename = "metrics_performance_chart.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filename)
        print(f"  Generated: {filename}")
        
        # 2. Confusion matrix
        plt.figure(figsize=(8, 6))
        tp, fp, fn, tn = metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']
        confusion_data = np.array([[tp, fp], [fn, tn]])
        
        im = plt.imshow(confusion_data, cmap='Blues')
        plt.title('Confusion Matrix Components', fontsize=14, fontweight='bold')
        plt.xticks([0, 1], ['Predicted\nFire', 'Predicted\nNo Fire'])
        plt.yticks([0, 1], ['Actual\nFire', 'Actual\nNo Fire'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f'{confusion_data[i, j]:.0f}', ha='center', va='center', 
                        fontsize=12, color='white' if confusion_data[i, j] > confusion_data.max()/2 else 'black')
        
        plt.colorbar(im, shrink=0.8)
        plt.tight_layout()
        filename = "metrics_confusion_matrix.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filename)
        print(f"  Generated: {filename}")
        
        # 3. Model configuration (as text file instead of graph)
        config_text = f"""Model Configuration:
Embed Dim: {self.model_config.get('embed_dim', 'N/A')}
Features: {self.model_config.get('num_features', 19)}
Max Height: {self.model_config.get('max_height', 64)}
Max Width: {self.model_config.get('max_width', 64)}
Batch Size: {self.model_config.get('batch_size', 'N/A')}
Fire Weight: {self.model_config.get('fire_weight', 'N/A')}
Dice Weight: {self.model_config.get('dice_weight', 'N/A')}"""
        
        filename = "model_configuration.txt"
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            f.write(config_text)
        generated_files.append(filename)
        print(f"  Generated: {filename}")
        
        print(f"  Sample Metrics - F1: {metrics['f1']:.3f}, IoU: {metrics['iou']:.3f}, "
              f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
        
        return generated_files, metrics
    
    def generate_sample_summary(self, sample_idx: int, metrics: Dict, 
                               feature_files: List[str], fire_files: List[str], 
                               metrics_files: List[str], output_dir: Path):
        """Generate JSON metrics and feature documentation"""
        
        # Generate metrics JSON
        metrics_data = {
            "sample_idx": sample_idx,
            "metrics": {
                "f1_score": float(metrics['f1']),
                "iou": float(metrics['iou']),
                "precision": float(metrics['precision']),
                "recall": float(metrics['recall']),
                "tp": float(metrics['tp']),
                "fp": float(metrics['fp']),
                "fn": float(metrics['fn']),
                "tn": float(metrics['tn'])
            }
        }
        
        metrics_file_path = output_dir / "sample_metrics.json"
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Generate feature documentation
        feature_doc = {
            "sample_idx": sample_idx,
            "dataset_info": {
                "name": "Enhanced Next Day Wildfire Spread (NDWS) Dataset",
                "paper": "An enhanced wildfire spread prediction using multimodal satellite imagery and deep learning models",
                "authors": "Rufai Yusuf Zakari, Owais Ahmed Malik, Ong Wee-Hong",
                "year": 2025,
                "spatial_resolution": "1 km",
                "temporal_coverage": "July 2015 to October 2024",
                "location": "Contiguous United States"
            },
            "features": {}
        }
        
        # Add feature information
        for i, feature_name in enumerate(ENHANCED_INPUT_FEATURES):
            feature_key = feature_name.lower().replace(' ', '').replace('(', '').replace(')', '')
            description = FEATURE_DESCRIPTIONS.get(feature_key, feature_name)
            
            # Find category
            category = "Unknown"
            for cat_name, cat_info in FEATURE_CATEGORIES.items():
                if feature_key in cat_info['features']:
                    category = cat_info['description']
                    break
            
            feature_doc["features"][f"feature_{i:02d}"] = {
                "name": feature_name,
                "description": description,
                "category": category,
                "filename": f"feature_{i:02d}_{feature_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
            }
        
        # Add fire progression information
        feature_doc["fire_progression"] = {
            "previous_fire": {
                "description": "Previous Day Fire Mask (Time t)",
                "filename": "fire_previous_fire.png",
                "meaning": "Shows fire locations from the previous day"
            },
            "ground_truth": {
                "description": "Ground Truth Fire Mask (Time t+1)", 
                "filename": "fire_ground_truth.png",
                "meaning": "Actual fire locations for the next day"
            },
            "prediction_probability": {
                "description": "Model Prediction (Probability)",
                "filename": "fire_prediction_probability.png", 
                "meaning": "Model's probability prediction for fire spread"
            },
            "prediction_binary": {
                "description": "Model Prediction (Binary Threshold > 0.5)",
                "filename": "fire_prediction_binary.png",
                "meaning": "Binary fire prediction using 0.5 threshold"
            },
            "comparison": {
                "description": "Fire Spread Comparison",
                "filename": "fire_comparison_overlay.png",
                "meaning": "Overlay showing prediction accuracy (Red=GT, Green=Pred, Yellow=Both)"
            }
        }
        
        feature_doc["metrics_performance_chart"] = {
            "description": "Performance metrics bar chart (Precision, Recall, F1, IoU)",
            "filename": "metrics_performance_chart.png"
        }
        feature_doc["metrics_confusion_matrix"] = {
            "description": "Confusion matrix visualization (TP, FP, FN, TN)",
            "filename": "metrics_confusion_matrix.png"
        }
        feature_doc["model_configuration"] = {
            "description": "Model configuration parameters",
            "filename": "model_configuration.txt"
        }
        
        doc_file_path = output_dir / "feature_documentation.json"
        with open(doc_file_path, 'w') as f:
            json.dump(feature_doc, f, indent=2)
        
        print(f"Generated metrics JSON: sample_metrics.json")
        print(f"Generated feature documentation: feature_documentation.json")
        return metrics_file_path, doc_file_path

def generate_single_sample(sample_idx: int, generator: SampleVisualizationGenerator, 
                          data_dir: str, base_output_dir: str = "visualizations"):
    """Generate visualizations for a single sample"""
    try:
        # Load test sample
        features, target = generator.load_test_sample(sample_idx, data_dir)
        
        # Create output directory
        output_dir = generator.create_sample_directory(sample_idx, base_output_dir)
        
        # Generate prediction
        prediction = generator.predict_sample(features)
        
        # Generate all visualizations
        feature_files = generator.generate_individual_feature_visualizations(features, output_dir)
        fire_files = generator.generate_fire_progression_visualization(
            features, target, prediction, output_dir)
        metrics_files, metrics = generator.generate_metrics_dashboard(prediction, target, output_dir)
        metrics_file_path, doc_file_path = generator.generate_sample_summary(
            sample_idx, metrics, feature_files, fire_files, metrics_files, output_dir)
        
        return {
            'sample_idx': sample_idx,
            'output_dir': output_dir,
            'metrics': metrics,
            'files_generated': len(feature_files) + len(fire_files) + 2
        }
        
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
        return None

def generate_single_sample_with_data(sample_idx: int, generator: SampleVisualizationGenerator, 
                                   features: np.ndarray, targets: np.ndarray, 
                                   base_output_dir: str = "visualizations"):
    """Generate visualizations for a single sample with pre-loaded data"""
    try:
        if sample_idx >= len(features):
            raise ValueError(f"Sample index {sample_idx} out of range. Available samples: {len(features)}")
            
        sample_features = features[sample_idx]
        sample_target = targets[sample_idx]
        
        print(f"Processing sample {sample_idx}")
        print(f"  Original features shape: {sample_features.shape}") 
        print(f"  Original target shape: {sample_target.shape}")
        
        # Store original for visualization
        generator.original_features = sample_features.copy()
        generator.original_target = sample_target.copy()
        
        # Preprocess for model inference
        processed_features = generator._preprocess_for_model(sample_features)
        processed_target = generator._preprocess_target_for_model(sample_target)
        
        print(f"  Processed features shape: {processed_features.shape}")
        print(f"  Processed target shape: {processed_target.shape}")
        
        # Create output directory
        output_dir = generator.create_sample_directory(sample_idx, base_output_dir)
        
        # Generate prediction
        prediction = generator.predict_sample(processed_features)
        
        # Upscale prediction back to original size for proper metrics calculation
        original_h, original_w = generator.original_target.shape[:2]
        
        # Handle different prediction shapes
        if len(prediction.shape) == 3:
            # 3D prediction (H, W, C) - provide zoom factors for all dimensions
            zoom_factors = (original_h/prediction.shape[0], original_w/prediction.shape[1], 1.0)
        else:
            # 2D prediction (H, W) - provide zoom factors for spatial dimensions only
            zoom_factors = (original_h/prediction.shape[0], original_w/prediction.shape[1])
        
        upscaled_prediction = zoom(prediction, zoom_factors, order=1)
        
        # Ensure upscaled prediction has same shape as original target
        if len(upscaled_prediction.shape) == 2:
            upscaled_prediction = upscaled_prediction[:, :, np.newaxis]
        
        print(f"  Upscaled prediction from {prediction.shape} to {upscaled_prediction.shape}")
        print(f"  Original target shape: {generator.original_target.shape}")
        
        # Generate all visualizations
        feature_files = generator.generate_individual_feature_visualizations(processed_features, output_dir)
        fire_files = generator.generate_fire_progression_visualization(
            sample_features, sample_target, upscaled_prediction, output_dir)
        # Use original prediction for metrics calculation (like local version)
        # Use upscaled prediction only for visualization
        metrics_files, metrics = generator.generate_metrics_dashboard(prediction, processed_target, output_dir)
        metrics_file_path, doc_file_path = generator.generate_sample_summary(
            sample_idx, metrics, feature_files, fire_files, metrics_files, output_dir)
        
        return {
            'sample_idx': sample_idx,
            'output_dir': output_dir,
            'metrics': metrics,
            'files_generated': len(feature_files) + len(fire_files) + 2
        }
        
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
        return None

def main(sample_idx: int = 0, data_dir: str = "data/processed", 
         model_path: str = "model_nfp.pth", generate_all: bool = False, max_samples: int = None):
    """Main function to generate visualizations for sample(s)
    
    Args:
        sample_idx: Index of the test sample to visualize (ignored if generate_all=True)
        data_dir: Directory containing processed data
        model_path: Path to the trained model
        generate_all: Generate all visualizations for all samples
        max_samples: Maximum number of samples to process (None = all samples)
    """
    
    try:
        # Initialize generator
        generator = SampleVisualizationGenerator(model_path)
        
        if generate_all:
            print("=" * 60)
            print("Generating Visualizations for ALL Test Samples")
            print("=" * 60)
            
            # This functionality is not supported in Railway deployment
            # Data must be loaded via Supabase and passed directly
            raise NotImplementedError("Batch processing is not supported in Railway deployment. Use individual sample processing instead.")
            
            total_samples = len(features)
            if max_samples is not None:
                total_samples = min(max_samples, total_samples)
            
            print(f"Processing {total_samples} samples...")
            
            # Track results
            successful_samples = []
            failed_samples = []
            all_metrics = []
            
            # Process each sample
            for idx in range(total_samples):
                print(f"\nProcessing sample {idx + 1}/{total_samples} (ID: {idx})")
                
                result = generate_single_sample(idx, generator, data_dir)
                
                if result:
                    successful_samples.append(result)
                    all_metrics.append({
                        'sample_idx': idx,
                        **result['metrics']
                    })
                    print(f"Sample {idx} completed - F1: {result['metrics']['f1']:.3f}, IoU: {result['metrics']['iou']:.3f}")
                else:
                    failed_samples.append(idx)
                    print(f"Sample {idx} failed")
                
                # Progress update every 10 samples
                if (idx + 1) % 10 == 0:
                    success_rate = len(successful_samples) / (idx + 1) * 100
                    print(f"\nProgress: {idx + 1}/{total_samples} ({success_rate:.1f}% success rate)")
            
            # Generate summary statistics
            generate_overall_summary(successful_samples, failed_samples, all_metrics)
            
            print("\n" + "=" * 60)
            print("BATCH PROCESSING COMPLETE!")
            print("=" * 60)
            print(f"Successfully processed: {len(successful_samples)}/{total_samples} samples")
            print(f"Failed samples: {len(failed_samples)}")
            if all_metrics:
                avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
                avg_iou = sum(m['iou'] for m in all_metrics) / len(all_metrics)
                print(f"Average F1: {avg_f1:.3f}, Average IoU: {avg_iou:.3f}")
            
            return successful_samples, failed_samples
            
        else:
            # Single sample processing (original functionality)
            print("=" * 60)
            print(f"Sample Visualization Generator")
            print(f"Sample Index: {sample_idx}")
            print("=" * 60)
            
            result = generate_single_sample(sample_idx, generator, data_dir)
            
            if result:
                print("\n" + "=" * 60)
                print("Visualization Generation Complete!")
                print("=" * 60)
                print(f"Output directory: {result['output_dir']}")
                print(f"Total files generated: {result['files_generated'] + 1}")  # +1 for feature documentation
                print(f"Sample metrics - F1: {result['metrics']['f1']:.3f}, IoU: {result['metrics']['iou']:.3f}")
                print(f"Metrics JSON: sample_metrics.json")
                print(f"Feature documentation: feature_documentation.json")
                
                return result['output_dir'], result['metrics']
            else:
                raise ValueError(f"Failed to process sample {sample_idx}")
        
    except Exception as e:
        print(f"Error in main processing: {e}")
        raise

def generate_overall_summary(successful_samples, failed_samples, all_metrics):
    """Generate overall summary statistics for batch processing"""
    
    # Create overall summary directory
    summary_dir = Path("visualizations/batch_summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary statistics
    summary_data = {
        "processing_summary": {
            "total_samples_attempted": len(successful_samples) + len(failed_samples),
            "successful_samples": len(successful_samples),
            "failed_samples": len(failed_samples),
            "success_rate": len(successful_samples) / (len(successful_samples) + len(failed_samples)) * 100
        },
        "failed_sample_indices": failed_samples,
        "metrics_summary": {}
    }
    
    if all_metrics:
        metrics_array = np.array([[m['f1'], m['iou'], m['precision'], m['recall']] for m in all_metrics])
        
        summary_data["metrics_summary"] = {
            "mean": {
                "f1_score": float(np.mean(metrics_array[:, 0])),
                "iou": float(np.mean(metrics_array[:, 1])),
                "precision": float(np.mean(metrics_array[:, 2])),
                "recall": float(np.mean(metrics_array[:, 3]))
            },
            "std": {
                "f1_score": float(np.std(metrics_array[:, 0])),
                "iou": float(np.std(metrics_array[:, 1])),
                "precision": float(np.std(metrics_array[:, 2])),
                "recall": float(np.std(metrics_array[:, 3]))
            },
            "min": {
                "f1_score": float(np.min(metrics_array[:, 0])),
                "iou": float(np.min(metrics_array[:, 1])),
                "precision": float(np.min(metrics_array[:, 2])),
                "recall": float(np.min(metrics_array[:, 3]))
            },
            "max": {
                "f1_score": float(np.max(metrics_array[:, 0])),
                "iou": float(np.max(metrics_array[:, 1])),
                "precision": float(np.max(metrics_array[:, 2])),  
                "recall": float(np.max(metrics_array[:, 3]))
            }
        }
    
    # Save overall summary
    with open(summary_dir / "batch_processing_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Save detailed metrics for all samples
    with open(summary_dir / "all_sample_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nBatch summary saved to: {summary_dir}")
    print(f"  - batch_processing_summary.json")
    print(f"  - all_sample_metrics.json")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive visualizations for a test sample')
    parser.add_argument('--sample-idx', type=int, default=0, 
                       help='Index of the test sample to visualize (default: 0)')
    parser.add_argument('--data-dir', type=str, default="data/processed",
                       help='Directory containing processed data (default: data/processed)')
    parser.add_argument('--model-path', type=str, default="model_nfp.pth",
                       help='Path to the trained model (default: model_nfp.pth)')
    parser.add_argument('--generate-all', action='store_true',
                       help='Generate all visualizations for all samples (default: False)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process when using --generate-all (default: None = all samples)')
    
    args = parser.parse_args()
    
    main(args.sample_idx, args.data_dir, args.model_path, args.generate_all, args.max_samples) 