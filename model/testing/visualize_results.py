import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob

# Allow modules to be imported from /model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ENHANCED_INPUT_FEATURES

from models import FlameAIModel
from train import load_ndws_data, calculate_segmentation_metrics
from interpretability import (
    GradCAM,
    IntegratedGradients,
    analyze_model_interpretability
)

class ModelVisualizer:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        """Load the trained model from checkpoint"""
        print(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.model_config = checkpoint.get('config', {})
        
        # Initialize model with saved config
        self.model = FlameAIModel(
            input_shape=(
                self.model_config.get('max_height', 64),
                self.model_config.get('max_width', 64), 
                self.model_config.get('num_features', 19)
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
        
    def load_test_data(self, data_dir="../data/processed", split="test", num_samples=None):
        """Load test data for visualization"""
        print("Loading test data...")
        
        # Try to load test data, fallback to train if test doesn't exist
        try:
            features, targets = load_ndws_data(data_dir, split)
        except:
            print(f"Could not load {split} data, using train data instead")
            features, targets = load_ndws_data(data_dir, "train")
        
        if features is None or targets is None:
            raise ValueError("Could not load data for visualization")
        
        # Limit samples if specified
        if num_samples:
            indices = np.random.choice(len(features), min(num_samples, len(features)), replace=False)
            features = features[indices]
            targets = targets[indices]
            
        print(f"Loaded {len(features)} samples for visualization")
        return features, targets
    
    def predict_batch(self, data_batch):
        """Run inference on a batch of data"""
        with torch.no_grad():
            if not isinstance(data_batch, torch.Tensor):
                data_batch = torch.Tensor(data_batch)
            data_batch = data_batch.to(self.device)
            predictions = self.model(data_batch)
            return predictions.cpu().numpy()
    
    def visualize_sample_predictions(self, data, targets, sample_indices=None, num_samples=5, save_path=None):
        """Visualize predictions for specific samples"""
        if sample_indices is None:
            sample_indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
        
        fig, axes = plt.subplots(len(sample_indices), 4, figsize=(16, 4*len(sample_indices)))

        # Make sure axes has the correct dimensions
        if len(sample_indices) == 1:
            axes = axes.reshape(1, -1)
            
        for i, idx in enumerate(sample_indices):
            sample_data = data[idx:idx+1]  # Keep batch dimension
            sample_target = targets[idx]
            
            # Get prediction
            prediction = self.predict_batch(sample_data)[0]  # Remove batch dimension
            
            # Previous fire mask (from input)
            prev_fire = sample_data[:, :, -1]
            
            # Ground truth
            ground_truth = sample_target[:, :, 0]
            
            # Prediction (squeeze if needed)
            pred_binary = (prediction[:, :, 0] > 0.5).astype(float)
            
            # Plot previous fire mask
            axes[i, 0].imshow(prev_fire, cmap='Reds', vmin=0, vmax=1)
            axes[i, 0].set_title(f'Sample {idx}\nPrevious Fire Mask')
            axes[i, 0].axis('off')
            
            # Plot ground truth
            axes[i, 1].imshow(ground_truth, cmap='Reds', vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth\n(Next Day)')
            axes[i, 1].axis('off')
            
            # Plot prediction
            axes[i, 2].imshow(prediction[:, :, 0], cmap='Reds', vmin=0, vmax=1)
            axes[i, 2].set_title('Prediction\n(Probability)')
            axes[i, 2].axis('off')
            
            # Plot comparison (overlay)
            comparison = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3))
            comparison[:, :, 0] = ground_truth  # Red for ground truth
            comparison[:, :, 1] = pred_binary   # Green for prediction
            comparison[:, :, 2] = 0
            
            axes[i, 3].imshow(comparison)
            axes[i, 3].set_title('Overlay\n(Red=GT, Green=Pred, Yellow=Both)')
            axes[i, 3].axis('off')
            
            # Calculate metrics for this sample
            sample_pred_tensor = torch.FloatTensor(prediction).unsqueeze(0)
            sample_target_tensor = torch.FloatTensor(sample_target).unsqueeze(0)
            metrics = calculate_segmentation_metrics(sample_pred_tensor, sample_target_tensor)
            
            # Add metrics as text
            metrics_text = f"F1: {metrics['f1']:.3f}, IoU: {metrics['iou']:.3f}\nPrecision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}"
            fig.text(0.02, 0.98 - i/len(sample_indices), metrics_text, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_feature_importance(self, data, sample_idx=0, save_path=None):
        """Visualize feature importance using Integrated Gradients"""
        print("Computing feature importance...")
        
        sample_data = data[sample_idx:sample_idx+1]
        
        # Initialize Integrated Gradients
        ig_analyzer = IntegratedGradients(self.model)
        importance_results = ig_analyzer.feature_importance_scores(
            torch.FloatTensor(sample_data).to(self.device)
        )
        
        # Create feature importance plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw importance scores
        feature_names = importance_results['feature_names']
        raw_scores = importance_results['raw_scores']
        
        bars1 = ax1.barh(range(len(feature_names)), raw_scores)
        ax1.set_yticks(range(len(feature_names)))
        ax1.set_yticklabels(feature_names, fontsize=10)
        ax1.set_xlabel('Raw Importance Score')
        ax1.set_title('Feature Importance (Raw Scores)')
        ax1.grid(axis='x', alpha=0.3)
        
        # Color bars by importance
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(raw_scores)))
        for bar, color in zip(bars1, colors):
            bar.set_color(color)
        
        # PCR scores
        pcr_scores = importance_results['pcr_scores']
        bars2 = ax2.barh(range(len(feature_names)), pcr_scores)
        ax2.set_yticks(range(len(feature_names)))
        ax2.set_yticklabels(feature_names, fontsize=10)
        ax2.set_xlabel('Positive Contribution Ratio')
        ax2.set_title('Feature Importance (Normalized)')
        ax2.grid(axis='x', alpha=0.3)
        
        # Color bars by importance
        for bar, color in zip(bars2, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_results
    
    def visualize_input_features(self, data, sample_idx=0, save_path=None):
        """Visualize all input features for a sample"""
        sample = data[sample_idx]
        
        num_features = len(ENHANCED_INPUT_FEATURES)

        # Create subplot grid
        cols = 4
        rows = (num_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []
        
        for i in range(num_features):
            feature_data = sample[:, :, i]
            
            im = axes[i].imshow(feature_data, cmap='viridis')
            axes[i].set_title(f'{ENHANCED_INPUT_FEATURES[i]}', fontsize=10)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(num_features, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Input Features - Sample {sample_idx}', fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_overall_metrics(self, data, targets, batch_size=8):
        """Calculate metrics across the entire dataset"""
        print("Calculating overall metrics...")
        
        all_predictions = []
        all_targets = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            predictions = self.predict_batch(batch_data)
            all_predictions.append(predictions)
            all_targets.append(batch_targets)
        
        # Concatenate all results
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        pred_tensor = torch.FloatTensor(all_predictions)
        target_tensor = torch.FloatTensor(all_targets)
        
        metrics = calculate_segmentation_metrics(pred_tensor, target_tensor)
        
        print("Overall Metrics:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"IoU: {metrics['iou']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        return metrics, all_predictions, all_targets
    
    def create_metrics_dashboard(self, metrics, save_path=None):
        """Create a dashboard showing model performance metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Metrics bar chart
        metric_names = ['Precision', 'Recall', 'F1 Score', 'IoU']
        metric_values = [metrics['precision'], metrics['recall'], metrics['f1'], metrics['iou']]
        
        bars = ax1.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Metrics')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Confusion matrix components
        tp, fp, fn, tn = metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']
        confusion_data = np.array([[tp, fp], [fn, tn]])
        
        im = ax2.imshow(confusion_data, cmap='Blues')
        ax2.set_title('Confusion Matrix Components')
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['Predicted Fire', 'Predicted No Fire'])
        ax2.set_yticklabels(['Actual Fire', 'Actual No Fire'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, f'{confusion_data[i, j]:.0f}', ha='center', va='center', fontsize=12)
        
        # Model configuration info
        config_text = f"""Model Configuration:
Embed Dim: {self.model_config.get('embed_dim', 'N/A')}
Batch Size: {self.model_config.get('batch_size', 'N/A')}  
Epochs: {self.model_config.get('epochs', 'N/A')}
Fire Weight: {self.model_config.get('fire_weight', 'N/A')}
Dice Weight: {self.model_config.get('dice_weight', 'N/A')}"""
        
        ax3.text(0.05, 0.95, config_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        ax3.set_title('Model Configuration')
        ax3.axis('off')
        
        # Performance comparison (if you have target values)
        target_metrics = {'F1 Score': 0.720, 'IoU': 0.635}  # From the paper
        current_metrics = {'F1 Score': metrics['f1'], 'IoU': metrics['iou']}
        
        x = np.arange(len(target_metrics))
        width = 0.35
        
        target_vals = list(target_metrics.values())
        current_vals = list(current_metrics.values())
        
        ax4.bar(x - width/2, target_vals, width, label='Paper Target', color='lightblue')
        ax4.bar(x + width/2, current_vals, width, label='Current Model', color='orange')
        
        ax4.set_ylabel('Score')
        ax4.set_title('Performance vs Target')
        ax4.set_xticks(x)
        ax4.set_xticklabels(target_metrics.keys())
        ax4.legend()
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Find available model files
    model_dir = "../models"
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    
    if not model_files:
        print(f"No model files found in {model_dir}/")
        return
    
    print("Available models:")
    for i, model_file in enumerate(model_files):
        print(f"{i}: {os.path.basename(model_file)}")
    
    model_path = model_files[0]  # Change this to select different models
    print(f"\nUsing model: {os.path.basename(model_path)}")
    
    # Initialize visualizer
    visualizer = ModelVisualizer(model_path)
    
    # Load test data
    try:
        data, targets = visualizer.load_test_data(num_samples=50)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create visualizations
    print("\n" + "="*50)
    print("Creating Visualizations")
    print("="*50)
    
    # 1. Sample predictions
    print("1. Visualizing sample predictions...")
    visualizer.visualize_sample_predictions(data, targets, num_samples=3, 
                                          save_path="../visualizations/sample_predictions.png")
    
    # 2. Input features
    print("2. Visualizing input features...")
    visualizer.visualize_input_features(data, sample_idx=0, save_path="visualizations/input_features.png")
    
    # 3. Feature importance
    print("3. Computing feature importance...")
    importance_results = visualizer.visualize_feature_importance(data, sample_idx=0,
                                                               save_path="../visualizations/feature_importance.png")
    
    # 4. Overall metrics
    print("4. Calculating overall metrics...")
    metrics, predictions, targets_array = visualizer.calculate_overall_metrics(data, targets)
    
    # 5. Metrics dashboard
    print("5. Creating metrics dashboard...")
    visualizer.create_metrics_dashboard(metrics, save_path="../visualizations/metrics_dashboard.png")
    
    print("\n" + "="*50)
    print("Visualization Complete!")
    print("="*50)
    print("Generated files:")
    print("- sample_predictions.png")
    print("- input_features_day0.png") 
    print("- feature_importance.png")
    print("- metrics_dashboard.png")

if __name__ == "__main__":
    main() 