import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import traceback
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from augmentation import create_random_crop_params, apply_random_crop, apply_rotation, check_valid_sample, add_surrounding_position

from models import FlameAIModel
from config import (
    ENHANCED_INPUT_FEATURES,
    NUM_ENHANCED_INPUT_FEATURES,
    DEFAULT_DATA_SIZE
)
# from interpretability import (
#     GradCAM,
#     IntegratedGradients,
#     analyze_model_interpretability,
#     visualize_interpretability_results
# )

def seed_everything(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    embed_dim = 128
    max_height = DEFAULT_DATA_SIZE  # 64 for NDWS
    max_width = DEFAULT_DATA_SIZE   # 64 for NDWS
    crop_size = 32  # Random crop size for augmentation
    batch_size = 16
    learning_rate = 0.004
    warmup_epochs = 1
    epochs = 10
    
    # Data augmentation settings
    use_random_crop = True
    use_rotation = True
    rotation_angles = [0, 90, 180, 270]  # Rotation angles for augmentation
    
    # Custom loss weights
    fire_weight = 10.0
    no_fire_weight = 1.0
    dice_weight = 2.0

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_device_info():
    """Print detailed device information"""
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check current GPU memory usage
        if torch.cuda.current_device() >= 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        print("No CUDA GPUs detected")
        print("Reasons might be:")
        print("  - No NVIDIA GPU available")
        print("  - CUDA not installed")
        print("  - PyTorch not compiled with CUDA support")

print_device_info()

# NDWS Data Loading Functions
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
        
        # Convert to (N, H, W, C) format expected by model
        features = np.transpose(features, (0, 2, 3, 1))  # (N, C, H, W) -> (N, H, W, C)
        labels = np.expand_dims(labels, axis=-1)  # (N, H, W) -> (N, H, W, 1)
        
        print(f"Loaded {split} data: {len(features)} samples")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")

        return features, labels
        
    except FileNotFoundError as e:
        print(f"Processed NDWS data not found: {e}")
        print("Please run data processing first:")
        print("  python run_data_processing.py --clean --data-dir data/raw --clean-output data/processed")
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

    # print(f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}, precision: {precision}, recall: {recall}')
    
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

class FlameDataset(Dataset):
    """Dataset class for NDWS wildfire prediction with augmentation"""
    def __init__(self, x, y, train=True, augment_factor=4):
        self.x = x  # Input features (N, H, W, features)
        self.y = y  # Target fire masks (N, H, W, 1)
        self.train = train
        self.augment_factor = augment_factor if train else 1
        
        # Pre-compute valid samples after potential cropping
        if train and Config.use_random_crop:
            self.valid_indices = self._find_valid_samples()
            print(f"Found {len(self.valid_indices)} valid samples for training after cropping filter")
        else:
            self.valid_indices = list(range(len(self.x)))
       
    def _find_valid_samples(self):
        """Find samples that have enough valid data after cropping"""
        valid_indices = []
        
        for i in range(len(self.x)):
            target = self.y[i]
            
            # Check multiple random crops to see if this sample can produce valid crops
            valid_crops = 0
            for _ in range(10):  # Check 10 random crop positions
                x_offset, y_offset = create_random_crop_params(target.shape[0], Config.crop_size)

                cropped_target = target[x_offset:x_offset+Config.crop_size, 
                                        y_offset:y_offset+Config.crop_size,
                                        :]
                
                if check_valid_sample(cropped_target, Config.crop_size):
                    valid_crops += 1
                    
            if valid_crops >= 3:  # At least 3 out of 10 crops should be valid
                valid_indices.append(i)
                
        return valid_indices
       
    def __len__(self):
        return len(self.valid_indices) * self.augment_factor
   
    def __getitem__(self, idx):
        # Map augmented index back to original sample
        original_idx = self.valid_indices[idx % len(self.valid_indices)]
        rotation_idx = idx // len(self.valid_indices) if self.train else 0
        
        x, y = self.x[original_idx].copy(), self.y[original_idx].copy()
       
        # Apply augmentations if training
        if self.train:
            # Random cropping
            if Config.use_random_crop:
                # Keep trying until we get a valid crop
                max_attempts = 20
                for attempt in range(max_attempts):
                    x_cropped, y_cropped = apply_random_crop(x, y, Config.crop_size)
                    if check_valid_sample(y_cropped, Config.crop_size):
                        x, y = x_cropped, y_cropped
                        break
                else:
                    # If we can't find a valid crop, use center crop
                    center_offset = (x.shape[1] - Config.crop_size) // 2
                    x = x[center_offset:center_offset+Config.crop_size, 
                          center_offset:center_offset+Config.crop_size, :]
                    y = y[center_offset:center_offset+Config.crop_size, 
                          center_offset:center_offset+Config.crop_size, :]
            
            # Rotation augmentation
            if Config.use_rotation and rotation_idx < len(Config.rotation_angles):
                angle = Config.rotation_angles[rotation_idx]
                x, y = apply_rotation(x, y, angle)
        
        # Convert to tensors before adding the surrounding position
        x = torch.Tensor(x)
        y = torch.Tensor(y)

        x = add_surrounding_position(x)

        # Move the tensors to the correct device
        return x.to(device), y.to(device)

class CustomWBCEDiceLoss(nn.Module):
    """
    Custom loss combining Weighted Binary Cross Entropy and Dice Loss
    for NDWS wildfire prediction with class imbalance handling
    """
    def __init__(self, w_fire=10.0, w_no_fire=1.0, dice_weight=2.0):
        super().__init__()
        self.w_fire = w_fire
        self.w_no_fire = w_no_fire
        self.dice_weight = dice_weight
        
    def forward(self, y_pred, y_true):
        # Handle invalid labels (-1) by creating a mask
        valid_mask = (y_true != -1.0)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            
        # Filter out invalid pixels
        y_pred_valid = y_pred[valid_mask]
        y_true_valid = y_true[valid_mask]
        
        # Weighted Binary Cross Entropy Loss
        weights = torch.where(y_true_valid == 1.0, self.w_fire, self.w_no_fire)
        wbce = F.binary_cross_entropy(y_pred_valid, y_true_valid, weight=weights)
        
        # Dice Loss
        intersection = (y_pred_valid * y_true_valid).sum()
        dice_loss = 1 - (2 * intersection + 1e-6) / (y_pred_valid.sum() + y_true_valid.sum() + 1e-6)
        
        # Combined loss
        total_loss = wbce + self.dice_weight * dice_loss
        
        return total_loss 

def train_model(model, train_loader, use_custom_loss=True):
    """Training function with custom loss and metrics for NDWS"""
    # Enable gradients
    model.train()
    
    # Use Adam optimizer as specified in NDWS paper
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    # Use linear scheduling with warmup
    warmup_steps = Config.warmup_epochs * len(train_loader)
    reduction_steps = (Config.epochs - Config.warmup_epochs) * len(train_loader)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, warmup_steps),
            torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.0, reduction_steps)
        ],
        milestones=[warmup_steps]
    )
    
    # Initialize custom loss function
    if use_custom_loss:
        criterion = CustomWBCEDiceLoss(
            w_fire=Config.fire_weight,
            w_no_fire=Config.no_fire_weight, 
            dice_weight=Config.dice_weight
        )
    else:
        criterion = nn.MSELoss()
   
    for epoch in range(Config.epochs):
        total_loss = 0
        total_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0}
        num_batches = 0
       
        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                optimizer.zero_grad()

                output = model(data)
                
                loss = criterion(output, target)
                
                # Calculate segmentation metrics
                metrics = calculate_segmentation_metrics(output.detach(), target.detach())
                
                # Check for NaN values
                if torch.isnan(loss):
                    print(f"Warning: NaN detected at epoch {epoch+1}, batch {batch_idx}")
                    continue
            
                loss.backward()
                        
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                optimizer.step()
            
                total_loss += loss.item()
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                num_batches += 1
            
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{Config.epochs}, Batch {batch_idx}/{len(train_loader)}, '
                                f'Loss: {loss.item():.6f}, F1: {metrics["f1"]:.4f}, '
                                f'IoU: {metrics["iou"]:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
                
                scheduler.step()
                            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                traceback.print_exc()
                continue
       
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_metrics = {key: val / num_batches for key, val in total_metrics.items()}
        
            print(f'Epoch {epoch+1}/{Config.epochs} - Avg Loss: {avg_loss:.6f}, '
                    f'Avg F1: {avg_metrics["f1"]:.4f}, Avg IoU: {avg_metrics["iou"]:.4f}, '
                    f'Avg Precision: {avg_metrics["precision"]:.4f}, Avg Recall: {avg_metrics["recall"]:.4f}')
        else:
            print(f'Epoch {epoch+1}/{Config.epochs} - No valid batches processed!')

# Main NDWS Training Loop
if __name__ == "__main__":
    print("=" * 60)
    print("FLAME AI Model Training for NDWS Dataset")
    print("=" * 60)
    
    # Load NDWS data
    print("Loading NDWS data...")
    features, targets = load_ndws_data("data/processed", "train")
    
    if features is None or targets is None:
        print("Failed to load NDWS data. Please ensure data is processed.")
        exit(1)

    print(f"\n" + "="*50)
    print(f"Starting Training")
    print("="*50)
    
    seed_everything(0)
    
    train_x = features
    train_y = targets
    
    # Calculate training steps
    train_steps = len(train_x) // Config.batch_size
    train_sample = train_steps * Config.batch_size

    # Trim data to fit batch size
    train_x = train_x[:train_sample]
    train_y = train_y[:train_sample]

    print(f"Training samples: {len(train_x)}")
    print(f"Training steps per epoch: {train_steps}")

    # Create data loader
    dataset = FlameDataset(train_x, train_y, train=True)

    train_loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for performance
    )
    
    # Build model - use crop size for model dimensions if cropping is enabled
    model_height = Config.crop_size if Config.use_random_crop else Config.max_height
    model_width = Config.crop_size if Config.use_random_crop else Config.max_width
    
    model = FlameAIModel(
        input_shape=(
            2,
            model_height,
            model_width,
            NUM_ENHANCED_INPUT_FEATURES * 9 # To account for surrounding position encoding
        ),
        embed_dim=Config.embed_dim,
        num_heads=8,
        attention_dropout=0.1,
        dropout=0.2
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on {len(train_loader)} batches")
    
    # Print augmentation settings
    print(f"\n📊 Data Augmentation Settings:")
    print(f"  Random Crop: {'ON' if Config.use_random_crop else 'OFF'} ({Config.crop_size}x{Config.crop_size})")
    print(f"  Rotation: {'ON' if Config.use_rotation else 'OFF'} {Config.rotation_angles}")
    print(f"  Augmentation Factor: {dataset.augment_factor}x")
    print(f"  Effective Dataset Size: {len(train_loader.dataset)} samples")
    
    # Train model
    train_model(model, train_loader, use_custom_loss=True)
    
    # Save model
    num_models = len([f for f in os.listdir('models') if f.endswith('.pth')])
    model_save_path = f'models/model_{num_models}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'embed_dim': Config.embed_dim,
            'max_height': Config.max_height,
            'max_width': Config.max_width,
            'crop_size': Config.crop_size,
            'batch_size': Config.batch_size,
            'epochs': Config.epochs,
            'num_features': NUM_ENHANCED_INPUT_FEATURES * 9,
            'fire_weight': Config.fire_weight,
            'dice_weight': Config.dice_weight,
            'use_random_crop': Config.use_random_crop,
            'use_rotation': Config.use_rotation,
            'rotation_angles': Config.rotation_angles,
        },
        'dataset_info': {
            'num_samples': len(train_x),
            'augmented_samples': len(train_loader.dataset),
            'input_shape': train_x.shape,
            'target_shape': train_y.shape,
            'features': ENHANCED_INPUT_FEATURES,
        }
    }, model_save_path)
    
    print(f"Model saved as {model_save_path}")
    
    # Clear memory
    del model, train_loader

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("FLAME AI Training Complete!")
    print("="*60) 