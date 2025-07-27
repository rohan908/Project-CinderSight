import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_ndws_data, create_temporal_sequence, preprocess_ndws
from models import FlameAIModel
from interpretability import calculate_segmentation_metrics

class ExistingModelComparator:
    """Compare two existing trained models on test data"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def load_model(self, model_path, model_name):
        """Load a trained model from .pth file with error handling"""
        print(f"Loading {model_name} from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Extract model dimensions from config or use defaults
        height = config.get('crop_size', config.get('max_height', 64))
        width = config.get('crop_size', config.get('max_width', 64))
        
        # Create model with saved configuration
        model = FlameAIModel(
            input_shape=(2, height, width, 15),
            embed_dim=config.get('embed_dim', 128),
            num_heads=8,
            attention_dropout=0.1,
            dropout=0.2
        ).to(self.device)
        
        # Load weights with error handling for architecture mismatches
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Model loaded successfully: {sum(p.numel() for p in model.parameters()):,} parameters")
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(f"  WARNING: Architecture mismatch detected for {model_name}")
                print("  Attempting to load with strict=False (some layers may not match)")
                
                # Try loading with strict=False to handle architecture changes
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint['model_state_dict'], 
                    strict=False
                )
                
                if missing_keys:
                    print(f"  Missing keys: {len(missing_keys)} layers")
                if unexpected_keys:
                    print(f"  Unexpected keys: {len(unexpected_keys)} layers")
                    
                print(f"  Model partially loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
                print("  NOTE: Results may not be accurate due to architecture mismatch")
            else:
                raise e
        
        model.eval()
        print(f"  Input size: {height}x{width}")
        
        return model, config
    
    def load_test_data(self):
        """Load test data for evaluation"""
        print("Loading test data...")
        
        # Try test split first, fall back to train if needed
        try:
            features, targets = load_ndws_data("../data/processed", "test")
            print("Using test split")
        except:
            try:
                features, targets = load_ndws_data("../data/processed", "train")
                # Use last 20% as test data
                split_idx = int(0.8 * len(features))
                features = features[split_idx:]
                targets = targets[split_idx:]
                print("Using last 20% of train data as test set")
            except Exception as e:
                raise ValueError(f"Could not load test data: {e}")
        
        # Create temporal sequences
        temporal_data, temporal_targets = create_temporal_sequence(features, targets)
        
        print(f"Test data loaded: {len(temporal_data)} samples")
        return temporal_data, temporal_targets
    
    def evaluate_model(self, model, data, targets, model_name, input_size):
        """Evaluate a model on test data with improved error handling"""
        print(f"Evaluating {model_name}...")
        
        all_predictions = []
        all_targets = []
        inference_times = []
        
        batch_size = 8
        successful_batches = 0
        
        with torch.no_grad():
            for i in range(0, min(len(data), 400), batch_size):  # Limit to 400 samples for speed
                try:
                    batch_data = data[i:i+batch_size]
                    batch_targets = targets[i:i+batch_size]
                    
                    # Crop data to match model input size if needed
                    if input_size < 64:
                        # Center crop to match model's expected input size
                        start = (64 - input_size) // 2
                        end = start + input_size
                        batch_data = batch_data[:, :, start:end, start:end, :]
                        batch_targets = batch_targets[:, start:end, start:end, :]
                    
                    # Process each sample individually to avoid batch issues
                    for j in range(len(batch_data)):
                        try:
                            x, y = preprocess_ndws(batch_data[j], batch_targets[j])
                            
                            import time
                            start_time = time.time()
                            pred = model(x.unsqueeze(0))  # Add batch dimension
                            inference_time = time.time() - start_time
                            inference_times.append(inference_time)
                            
                            # Ensure output dimensions match target dimensions
                            if pred.shape != y.unsqueeze(0).shape:
                                # Resize prediction to match target if needed
                                pred = torch.nn.functional.interpolate(
                                    pred, 
                                    size=y.shape[1:3], 
                                    mode='bilinear', 
                                    align_corners=False
                                )
                            
                            all_predictions.append(pred.cpu())
                            all_targets.append(y.unsqueeze(0).cpu())
                            
                        except Exception as sample_error:
                            print(f"  Warning: Skipping sample {j} due to error: {sample_error}")
                            continue
                    
                    successful_batches += 1
                    
                except Exception as batch_error:
                    print(f"  Warning: Skipping batch {i//batch_size} due to error: {batch_error}")
                    continue
        
        if not all_predictions:
            raise ValueError("No valid predictions generated - model may be incompatible")
        
        print(f"  Successfully processed {len(all_predictions)} samples from {successful_batches} batches")
        
        # Calculate metrics with improved error handling
        try:
            predictions = torch.cat(all_predictions, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)
            
            # Simple metrics calculation to avoid complex shape issues
            predictions_flat = predictions.view(-1)
            targets_flat = targets_tensor.view(-1)
            
            # Create valid mask (exclude -1 values)
            valid_mask = targets_flat != -1.0
            
            if valid_mask.sum() == 0:
                print("  Warning: No valid target pixels found")
                metrics = {'f1': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
            else:
                valid_preds = predictions_flat[valid_mask]
                valid_targets = targets_flat[valid_mask]
                
                # Apply threshold
                threshold = 0.5
                pred_binary = (valid_preds > threshold).float()
                target_binary = valid_targets.float()
                
                # Calculate basic metrics
                tp = (pred_binary * target_binary).sum().item()
                fp = (pred_binary * (1 - target_binary)).sum().item()
                fn = ((1 - pred_binary) * target_binary).sum().item()
                tn = ((1 - pred_binary) * (1 - target_binary)).sum().item()
                
                # Calculate metrics with division by zero protection
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                
                metrics = {
                    'f1': f1,
                    'iou': iou,
                    'precision': precision,
                    'recall': recall
                }
        
        except Exception as metrics_error:
            print(f"  Warning: Error calculating metrics: {metrics_error}")
            metrics = {'f1': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Calculate average inference time
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        
        results = {
            'f1': metrics['f1'],
            'iou': metrics['iou'], 
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'samples_evaluated': len(all_predictions),
            'avg_inference_time_ms': avg_inference_time * 1000,
            'input_size': f"{input_size}x{input_size}"
        }
        
        print(f"  Samples evaluated: {len(all_predictions)}")
        print(f"  F1: {results['f1']:.4f}")
        print(f"  IoU: {results['iou']:.4f}")
        print(f"  Avg inference: {results['avg_inference_time_ms']:.1f}ms")
        
        return results
    
    def compare_models(self, model1_path, model2_path, model1_name="Model 1", model2_name="Model 2"):
        """Compare two existing models"""
        print("="*60)
        print("COMPARING EXISTING TRAINED MODELS")
        print("="*60)
        
        # Load models
        model1, config1 = self.load_model(model1_path, model1_name)
        model2, config2 = self.load_model(model2_path, model2_name)
        
        # Load test data
        test_data, test_targets = self.load_test_data()
        
        # Get input sizes
        input_size1 = config1.get('crop_size', config1.get('max_height', 64))
        input_size2 = config2.get('crop_size', config2.get('max_width', 64))
        
        # Evaluate both models
        results1 = self.evaluate_model(model1, test_data, test_targets, model1_name, input_size1)
        results2 = self.evaluate_model(model2, test_data, test_targets, model2_name, input_size2)
        
        # Store results
        self.results = {
            model1_name: results1,
            model2_name: results2,
            'model1_config': config1,
            'model2_config': config2
        }
        
        # Generate comparison
        self.print_comparison(model1_name, model2_name)
        self.save_results(model1_name, model2_name)
        self.plot_comparison(model1_name, model2_name)
        
        return self.results
    
    def print_comparison(self, model1_name, model2_name):
        """Print detailed comparison"""
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        results1 = self.results[model1_name]
        results2 = self.results[model2_name]
        
        print(f"{'Metric':<15} {model1_name:<12} {model2_name:<12} {'Change':<12}")
        print("-" * 55)
        
        for metric in ['f1', 'iou', 'precision', 'recall']:
            val1 = results1[metric]
            val2 = results2[metric]
            change = ((val2 - val1) / val1) * 100 if val1 > 0 else 0
            
            status = "IMPROVED" if change > 0 else "DECLINED" if change < 0 else "SAME"
            print(f"{metric:<15} {val1:<12.4f} {val2:<12.4f} {status} {change:+6.2f}%")
        
        print(f"\nPerformance Summary:")
        print(f"  {model1_name}: {results1['samples_evaluated']} samples, {results1['input_size']}")
        print(f"  {model2_name}: {results2['samples_evaluated']} samples, {results2['input_size']}")
        
        print(f"\nInference Speed:")
        print(f"  {model1_name}: {results1['avg_inference_time_ms']:.1f}ms")
        print(f"  {model2_name}: {results2['avg_inference_time_ms']:.1f}ms")
        
        # Overall assessment
        f1_improvement = ((results2['f1'] - results1['f1']) / results1['f1']) * 100 if results1['f1'] > 0 else 0
        if f1_improvement > 5:
            print(f"\nASSESSMENT: {model2_name} shows significant improvement!")
        elif f1_improvement > 0:
            print(f"\nASSESSMENT: {model2_name} shows modest improvement")
        else:
            print(f"\nASSESSMENT: {model1_name} performs better or similarly")
    
    def save_results(self, model1_name, model2_name):
        """Save comparison results to JSON"""
        results_with_metadata = {
            'timestamp': datetime.now().isoformat(),
            'comparison_type': 'existing_models',
            'models_compared': [model1_name, model2_name],
            'results': self.results
        }
        
        filename = 'existing_models_comparison.json'
        with open(filename, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def plot_comparison(self, model1_name, model2_name):
        """Create comparison visualization"""
        results1 = self.results[model1_name]
        results2 = self.results[model2_name]
        
        metrics = ['f1', 'iou', 'precision', 'recall']
        values1 = [results1[m] for m in metrics]
        values2 = [results2[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance comparison
        bars1 = ax1.bar(x - width/2, values1, width, label=model1_name, color='skyblue')
        bars2 = ax1.bar(x + width/2, values2, width, label=model2_name, color='orange')
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (v1, v2) in enumerate(zip(values1, values2)):
            ax1.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom')
            ax1.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom')
        
        # Inference speed comparison
        speed1 = results1['avg_inference_time_ms']
        speed2 = results2['avg_inference_time_ms']
        
        ax2.bar([model1_name, model2_name], [speed1, speed2], color=['skyblue', 'orange'])
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Inference Speed Comparison')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        ax2.text(0, speed1 + max(speed1, speed2) * 0.02, f'{speed1:.1f}ms', ha='center', va='bottom')
        ax2.text(1, speed2 + max(speed1, speed2) * 0.02, f'{speed2:.1f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('existing_models_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison chart saved as existing_models_comparison.png")

def main():
    """Command-line interface for comparing existing models"""
    if len(sys.argv) != 3:
        print("Usage: python compare_existing_models.py <model1.pth> <model2.pth>")
        print("\nExample:")
        print("  python compare_existing_models.py ../models/old_model.pth ../models/new_model.pth")
        return
    
    model1_path = sys.argv[1]
    model2_path = sys.argv[2]
    
    # Extract model names from paths
    model1_name = os.path.splitext(os.path.basename(model1_path))[0]
    model2_name = os.path.splitext(os.path.basename(model2_path))[0]
    
    comparator = ExistingModelComparator()
    
    try:
        results = comparator.compare_models(model1_path, model2_path, model1_name, model2_name)
        print("\nComparison completed successfully!")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 