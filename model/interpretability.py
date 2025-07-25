import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from config import ENHANCED_INPUT_FEATURES

# Interpretability Methods for NDWS
import shap
try:
    import captum
    from captum.attr import IntegratedGradients, GradientShap
    CAPTUM_AVAILABLE = True
except ImportError:
    print("Warning: Captum not available. Install with: pip install captum")
    CAPTUM_AVAILABLE = False

class GradCAM:
    """Grad-CAM implementation for NDWS wildfire prediction"""
    def __init__(self, model, target_layer_name='cnn.final_conv_block'):
        self.model = model
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # Register hooks for the target layer
        for name, module in model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                module.register_forward_hook(self.save_activation)
                module.register_backward_hook(self.save_gradient)
                break
                
        if self.target_layer is None:
            print(f"Warning: Target layer {target_layer_name} not found")
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for wildfire prediction"""
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            # For fire prediction, focus on fire class (positive pixels)
            target_class = (output > 0.5).float().sum()
        
        # Backward pass
        self.model.zero_grad()
        target_class.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            print("Warning: Gradients or activations not captured")
            return None
            
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        
        # Resize to input size
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                           size=input_tensor.shape[2:4], mode='bilinear', align_corners=False)
        
        return cam.squeeze()

class IntegratedGradients:
    """Integrated Gradients for NDWS feature attribution"""
    def __init__(self, model):
        self.model = model
        
    def generate_baseline(self, input_tensor):
        """Generate baseline (zeros) for integrated gradients"""
        return torch.zeros_like(input_tensor)
        
    def compute_gradients(self, input_tensor):
        """Compute gradients of model output w.r.t. input"""
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        # Focus on fire prediction (sum of positive predictions)
        target = output.sum()
        
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        return input_tensor.grad.clone()
        
    def integrated_gradients(self, input_tensor, steps=50):
        """Compute integrated gradients attribution"""
        baseline = self.generate_baseline(input_tensor)
        
        # Generate interpolated inputs along straight line path
        alphas = torch.linspace(0, 1, steps, device=input_tensor.device)
        integrated_grads = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (input_tensor - baseline)
            
            # Compute gradients
            grads = self.compute_gradients(interpolated)
            integrated_grads += grads
            
        # Scale by path and average over steps
        integrated_grads *= (input_tensor - baseline) / steps
        
        return integrated_grads
        
    def feature_importance_scores(self, input_tensor):
        """Calculate feature importance scores for NDWS features"""
        ig_attributions = self.integrated_gradients(input_tensor)
        
        # Sum attributions across spatial dimensions for each feature
        feature_scores = torch.mean(ig_attributions, dim=(0, 2, 3))  # Average over batch and spatial dims
        
        # Calculate positive contribution ratio (PCR) for each feature
        positive_scores = torch.clamp(feature_scores, min=0)
        total_positive = positive_scores.sum()
        
        if total_positive > 0:
            pcr_scores = positive_scores / total_positive
        else:
            pcr_scores = torch.zeros_like(positive_scores)
            
        return {
            'raw_scores': feature_scores.cpu().numpy(),
            'pcr_scores': pcr_scores.cpu().numpy(),
            'feature_names': ENHANCED_INPUT_FEATURES[:len(feature_scores)]
        }

def analyze_model_interpretability(model, test_data, feature_names=None):
    """Comprehensive interpretability analysis for NDWS model"""
    print("Starting interpretability analysis...")
    
    results = {}
    
    # 1. Grad-CAM Analysis
    print("Generating Grad-CAM visualizations...")
    grad_cam = GradCAM(model)
    cam_maps = []
    
    for i, sample in enumerate(test_data[:5]):  # Analyze first 5 samples
        cam = grad_cam.generate_cam(sample.unsqueeze(0))
        if cam is not None:
            cam_maps.append(cam.cpu().numpy())
        if i >= 4:  # Limit to 5 samples
            break
            
    results['grad_cam'] = cam_maps
    
    # 2. Integrated Gradients Analysis  
    print("Computing Integrated Gradients...")
    ig_analyzer = IntegratedGradients(model)
    
    feature_importance_list = []
    for i, sample in enumerate(test_data[:10]):  # Analyze first 10 samples
        importance = ig_analyzer.feature_importance_scores(sample.unsqueeze(0))
        feature_importance_list.append(importance)
        if i >= 9:  # Limit to 10 samples
            break
            
    results['integrated_gradients'] = feature_importance_list
    
    # 3. SHAP Analysis (if available)
    if CAPTUM_AVAILABLE:
        print("Computing SHAP values...")
        try:
            # Use GradientShap from Captum
            gradient_shap = GradientShap(model)
            
            # Select a subset for baseline
            baselines = test_data[:5]
            
            shap_values_list = []
            for i, sample in enumerate(test_data[:5]):
                shap_vals = gradient_shap.attribute(
                    sample.unsqueeze(0), 
                    baselines=baselines,
                    n_samples=50
                )
                shap_values_list.append(shap_vals.cpu().numpy())
                
            results['shap'] = shap_values_list
            
        except Exception as e:
            print(f"SHAP analysis failed: {str(e)}")
            results['shap'] = None
    else:
        results['shap'] = None
    
    print("Interpretability analysis complete!")
    return results

def visualize_interpretability_results(results, save_path="interpretability_results.png"):
    """Visualize interpretability analysis results"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Grad-CAM visualization
    if results.get('grad_cam'):
        axes[0, 0].imshow(results['grad_cam'][0], cmap='hot')
        axes[0, 0].set_title('Grad-CAM Heatmap')
        axes[0, 0].axis('off')
    
    # Feature importance from IG
    if results.get('integrated_gradients'):
        importance = results['integrated_gradients'][0]
        axes[0, 1].bar(range(len(importance['pcr_scores'])), importance['pcr_scores'])
        axes[0, 1].set_title('Feature Importance (IG)')
        axes[0, 1].set_xlabel('Features')
        axes[0, 1].set_ylabel('Importance Score')
    
    # SHAP visualization
    if results.get('shap') and results['shap'] is not None:
        shap_mean = np.mean(results['shap'][0], axis=(0, 2, 3))
        axes[0, 2].bar(range(len(shap_mean)), shap_mean)
        axes[0, 2].set_title('SHAP Feature Importance')
        axes[0, 2].set_xlabel('Features')
        axes[0, 2].set_ylabel('SHAP Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# NDWS Evaluation Metrics
def calculate_segmentation_metrics(y_pred, y_true, threshold=0.5):
    """
    Calculate segmentation metrics for NDWS wildfire prediction
    """
    # Handle invalid labels
    valid_mask = (y_true != -1.0)
    
    if not valid_mask.any():
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0}
    
    y_pred_valid = (y_pred[valid_mask] > threshold).float()
    y_true_valid = y_true[valid_mask]
    
    # Calculate metrics
    tp = (y_pred_valid * y_true_valid).sum().item()
    fp = (y_pred_valid * (1 - y_true_valid)).sum().item()
    fn = ((1 - y_pred_valid) * y_true_valid).sum().item()
    tn = ((1 - y_pred_valid) * (1 - y_true_valid)).sum().item()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall, 
        'f1': f1,
        'iou': iou,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    } 