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
    """Calculate segmentation metrics for binary classification"""
    try:
        # Convert to binary predictions
        y_pred_binary = (y_pred > threshold).float()
        
        # Flatten tensors for calculation
        y_pred_flat = y_pred_binary.view(-1)
        y_true_flat = y_true.view(-1)
        
        # Calculate confusion matrix components
        tp = torch.sum((y_pred_flat == 1) & (y_true_flat == 1)).float()
        fp = torch.sum((y_pred_flat == 1) & (y_true_flat == 0)).float()
        fn = torch.sum((y_pred_flat == 0) & (y_true_flat == 1)).float()
        tn = torch.sum((y_pred_flat == 0) & (y_true_flat == 0)).float()
        
        # Calculate metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'iou': iou.item(),
            'tp': tp.item(),
            'fp': fp.item(),
            'fn': fn.item(),
            'tn': tn.item()
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'iou': 0.0,
            'tp': 0.0,
            'fp': 0.0,
            'fn': 0.0,
            'tn': 0.0
        } 