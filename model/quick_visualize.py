import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob

from config import ENHANCED_INPUT_FEATURES
from models import FlameAIModel
from train import load_ndws_data, create_temporal_sequence, calculate_segmentation_metrics

def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a trained model"""
    device = torch.device(device)
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Initialize model with correct dimensions
    model_height = config.get('crop_size', config.get('max_height', 64))
    model_width = config.get('crop_size', config.get('max_width', 64))
    
    model = FlameAIModel(
        input_shape=(2, model_height, model_width, 15),
        embed_dim=config.get('embed_dim', 128),
        num_heads=8,
        attention_dropout=0.1,
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, config, device

def quick_predict(model, data_sample, device):
    """Make prediction on a single sample"""
    with torch.no_grad():
        if not isinstance(data_sample, torch.Tensor):
            data_sample = torch.FloatTensor(data_sample)
        data_sample = data_sample.unsqueeze(0).to(device)  # Add batch dimension
        prediction = model(data_sample)
        return prediction.cpu().numpy()[0]  # Remove batch dimension

def visualize_predictions(model, data, targets, device, num_samples=3):
    """Show predictions vs ground truth"""
    indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        sample_data = data[idx]
        sample_target = targets[idx]
        
        # Get prediction
        prediction = quick_predict(model, sample_data, device)
        
        # Extract key data
        prev_fire = sample_data[0, :, :, -1]  # Previous fire mask
        ground_truth = sample_target[:, :, 0]
        pred_prob = prediction[:, :, 0]
        pred_binary = (pred_prob > 0.5).astype(float)
        
        # Calculate metrics
        sample_pred = torch.FloatTensor(prediction).unsqueeze(0)
        sample_tgt = torch.FloatTensor(sample_target).unsqueeze(0)
        metrics = calculate_segmentation_metrics(sample_pred, sample_tgt)
        
        # Plot
        axes[i, 0].imshow(prev_fire, cmap='Reds', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Previous Fire\n(Sample {idx})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(ground_truth, cmap='Reds', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_prob, cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Prediction\nF1: {metrics["f1"]:.3f}')
        axes[i, 2].axis('off')
        
        # Overlay comparison
        overlay = np.zeros((*ground_truth.shape, 3))
        overlay[:, :, 0] = ground_truth  # Red = GT
        overlay[:, :, 1] = pred_binary   # Green = Prediction
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f'Overlay\nIoU: {metrics["iou"]:.3f}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/quick_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def show_input_features(data, sample_idx=0):
    """Show input features for a sample"""
    sample = data[sample_idx]
    day0_data = sample[0]  # Current day features
    
    # Show key features
    key_features = ['vs', 'pr', 'tmmx', 'tmmn', 'NDVI', 'elevation', 'PrevFireMask']
    feature_indices = []
    
    for feat in key_features:
        if feat in ENHANCED_INPUT_FEATURES:
            feature_indices.append(ENHANCED_INPUT_FEATURES.index(feat))
    
    # Limit to available features
    available_indices = [i for i in feature_indices if i < day0_data.shape[-1]]
    available_features = [key_features[feature_indices.index(i)] for i in available_indices]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (feat_idx, feat_name) in enumerate(zip(available_indices, available_features)):
        if i < 8:  # Limit to subplot grid
            feature_data = day0_data[:, :, feat_idx]
            im = axes[i].imshow(feature_data, cmap='viridis')
            axes[i].set_title(feat_name)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    # Hide unused subplots
    for i in range(len(available_indices), 8):
        axes[i].axis('off')
    
    plt.suptitle(f'Key Input Features - Sample {sample_idx}')
    plt.tight_layout()
    plt.savefig('visualizations/input_features.png', dpi=150, bbox_inches='tight')
    plt.show()

def calculate_metrics(model, data, targets, device, batch_size=4):
    """Calculate overall metrics"""
    print("Calculating metrics...")
    
    all_preds = []
    all_targets = []
    
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        
        # Process batch
        batch_preds = []
        for sample in batch_data:
            pred = quick_predict(model, sample, device)
            batch_preds.append(pred)
        
        all_preds.extend(batch_preds)
        all_targets.extend(batch_targets)
    
    # Convert to tensors
    pred_tensor = torch.FloatTensor(np.array(all_preds))
    target_tensor = torch.FloatTensor(np.array(all_targets))
    
    metrics = calculate_segmentation_metrics(pred_tensor, target_tensor)
    
    print(f"Overall Results:")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    return metrics

def main():
    print("FLAME AI Model Visualization")
    print("="*50)
    
    # Find model files
    model_files = glob.glob("models/*.pth")
    if not model_files:
        print("No model files found in models/ directory!")
        return
    
    print("Available models:")
    for i, model_file in enumerate(model_files):
        print(f"  {i}: {os.path.basename(model_file)}")
    
    # Use first model (or modify to choose)
    model_path = model_files[0]
    print(f"\nUsing: {os.path.basename(model_path)}")
    
    # Load model
    model, config, device = load_model(model_path)
    
    # Load data
    print("\nLoading data...")
    try:
        # Try test data first, fallback to train
        try:
            features, targets = load_ndws_data("data/processed", "test")
        except:
            print("No test data found, using train data...")
            features, targets = load_ndws_data("data/processed", "train")
        
        temporal_data, temporal_targets = create_temporal_sequence(features, targets)
        
        # Use subset for visualization
        num_samples = min(20, len(temporal_data))
        data = temporal_data[:num_samples]
        targets = temporal_targets[:num_samples]
        
        print(f"Loaded {len(data)} samples")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("\n" + "="*50)
    print("Creating Visualizations...")
    print("="*50)
    
    # 1. Show predictions
    print("1. Generating prediction comparisons...")
    visualize_predictions(model, data, targets, device, num_samples=3)
    
    # 2. Show input features
    print("2. Showing input features...")
    show_input_features(data, sample_idx=0)
    
    # 3. Calculate metrics
    print("3. Calculating overall metrics...")
    metrics = calculate_metrics(model, data, targets, device)
    
    # 4. Show metrics comparison
    print("\n" + "="*30)
    print("📊 PERFORMANCE COMPARISON")
    print("="*30)
    print(f"{'Metric':<12} {'Current':<10} {'Target':<10} {'Difference':<10}")
    print("-" * 42)
    
    targets_paper = {'F1': 0.720, 'IoU': 0.635}
    current = {'F1': metrics['f1'], 'IoU': metrics['iou']}
    
    for metric in ['F1', 'IoU']:
        diff = current[metric] - targets_paper[metric]
        print(f"{metric:<12} {current[metric]:<10.3f} {targets_paper[metric]:<10.3f} {diff:+.3f}")
    
    print("\nVisualization complete!")
    print("Generated files:")
    print("  - quick_predictions.png")
    print("  - input_features.png")

if __name__ == "__main__":
    main() 