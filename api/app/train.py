import os
import numpy as np
import torch
import pickle

def load_ndws_data(data_dir="data/processed", split="train"):
    """Load NDWS data from processed pickle files"""
    try:
        # Load features (19 input channels: weather, terrain, vegetation, human, PrevFireMask)
        features_path = os.path.join(data_dir, f'{split}.data')
        with open(features_path, 'rb') as f:
            features = pickle.load(f)  # Shape: (N, C, H, W)
        
        # Load labels (FireMask - what we want to predict) 
        labels_path = os.path.join(data_dir, f'{split}.labels')
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)  # Shape: (N, H, W)
        
        print(f"Loaded {split} data:")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        return features, labels
        
    except Exception as e:
        print(f"Error loading {split} data: {e}")
        return None, None

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
        'tp': tp, # True Positives
        'fp': fp, # False Positives
        'fn': fn, # False Negatives
        'tn': tn # True Negatives
    } 