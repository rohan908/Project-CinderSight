import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import random
import math

# Import NDWS configuration
from config import (
    ENHANCED_INPUT_FEATURES, 
    OUTPUT_FEATURES, 
    ENHANCED_DATA_STATS,
    NUM_ENHANCED_INPUT_CHANNELS,
    DEFAULT_DATA_SIZE,
    WEATHER_CURRENT_FEATURES,
    WEATHER_FORECAST_FEATURES,
    TERRAIN_FEATURES,
    VEGETATION_FEATURES,
    HUMAN_FEATURES,
    FIRE_FEATURES
)

# Import models and interpretability functions
from models import (
    FlameAIModel,
    CustomWBCEDiceLoss,
    positional_encoding
)
from interpretability import (
    GradCAM,
    IntegratedGradients,
    analyze_model_interpretability,
    visualize_interpretability_results,
    calculate_segmentation_metrics
)

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
    batch_size = 64  # Increased from 16 to match paper
    epochs = 10  # Reduced from 20 to 10
    temporal_sequence = 2  # 2-day sequence: current day + next day
    awp_lambda = 0.04
    num_awp_epoch = 5
    
    # NDWS specific
    day0_features = len(WEATHER_CURRENT_FEATURES + TERRAIN_FEATURES + 
                       VEGETATION_FEATURES + HUMAN_FEATURES + FIRE_FEATURES)  # 15
    day1_input_features = len(WEATHER_FORECAST_FEATURES + TERRAIN_FEATURES + 
                             VEGETATION_FEATURES + HUMAN_FEATURES)  # 10 (no FireMask for input)
    max_features_per_day = max(day0_features, day1_input_features)  # 15
    
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
        print("  - No NVIDIA GPU installed")
        print("  - CUDA not installed")
        print("  - PyTorch not compiled with CUDA support")

print_device_info()

# NDWS Data Loading Functions
def load_ndws_data(data_dir="data/processed", split="train"):
    """Load NDWS data from processed pickle files"""
    import pickle
    
    # Load preprocessed data (as saved by data_cleaner.py)
    try:
        # Load features (19 input channels: weather, terrain, vegetation, human, PrevFireMask)
        features_path = os.path.join(data_dir, f'{split}.data')
        with open(features_path, 'rb') as f:
            features = pickle.load(f)  # Shape: (N, 19, H, W)
        
        # Load labels (FireMask - what we want to predict) 
        labels_path = os.path.join(data_dir, f'{split}.labels')
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)  # Shape: (N, H, W)
        
        # Convert to (N, H, W, C) format expected by model
        features = np.transpose(features, (0, 2, 3, 1))  # (N, 19, H, W) -> (N, H, W, 19)
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

def create_temporal_sequence(features, targets):
    """
    Create 2-day temporal sequences from NDWS data
    
    Args:
        features: Array of shape (N, H, W, 19) with all features
        targets: Array of shape (N, H, W, 1) with FireMask targets
        
    Returns:
        temporal_data: Array of shape (N, 2, H, W, 15) for 2-day sequences
        temporal_targets: Array of shape (N, H, W, 1) for next-day targets
    """
    N, H, W, _ = features.shape
    
    # Feature indices based on config.py
    current_weather_idx = [ENHANCED_INPUT_FEATURES.index(f) for f in WEATHER_CURRENT_FEATURES]
    forecast_weather_idx = [ENHANCED_INPUT_FEATURES.index(f) for f in WEATHER_FORECAST_FEATURES] 
    terrain_idx = [ENHANCED_INPUT_FEATURES.index(f) for f in TERRAIN_FEATURES]
    vegetation_idx = [ENHANCED_INPUT_FEATURES.index(f) for f in VEGETATION_FEATURES]
    human_idx = [ENHANCED_INPUT_FEATURES.index(f) for f in HUMAN_FEATURES]
    fire_idx = [ENHANCED_INPUT_FEATURES.index(f) for f in FIRE_FEATURES]
    
    # Create temporal sequences
    temporal_data = np.zeros((N, 2, H, W, Config.max_features_per_day))
    
    # Day 0 (current day): All available features except forecasts
    day0_features = (current_weather_idx + terrain_idx + vegetation_idx + 
                     human_idx + fire_idx)  # 15 features
    temporal_data[:, 0, :, :, :len(day0_features)] = features[:, :, :, day0_features]
    
    # Day 1 (next day): Forecasts + static features (no FireMask as input)
    day1_input_features = (forecast_weather_idx + terrain_idx + vegetation_idx + 
                          human_idx)  # 10 features
    day1_data = np.concatenate([
        features[:, :, :, forecast_weather_idx],  # Forecast weather
        features[:, :, :, terrain_idx],           # Terrain (static)
        features[:, :, :, vegetation_idx],        # Vegetation (static)
        features[:, :, :, human_idx]              # Human (static)
    ], axis=-1)
    temporal_data[:, 1, :, :, :len(day1_input_features)] = day1_data
    
    return temporal_data, targets

def add_surrounding_position(array):
    """
    Add surrounding position encoding for spatial context
    
    Args:
        array: Shape (batch_size, time_steps, height, width, channels)
    
    Returns:
        expanded_array: Shape (batch_size, time_steps, height, width, expanded_channels)
    """
    batch_size, time_steps, height, width, channels = array.shape

    # Padding the array: pad along the height and width dimensions only
    padded_array = F.pad(array, (0, 0, 1, 1, 1, 1, 0, 0, 0, 0), mode='constant')

    # Extract surrounding pixels
    center = array
    up = padded_array[:, :, :-2, 1:-1, :]
    down = padded_array[:, :, 2:, 1:-1, :]
    left = padded_array[:, :, 1:-1, :-2, :]
    right = padded_array[:, :, 1:-1, 2:, :]
    up_left = padded_array[:, :, :-2, :-2, :]
    up_right = padded_array[:, :, :-2, 2:, :]
    down_left = padded_array[:, :, 2:, :-2, :]
    down_right = padded_array[:, :, 2:, 2:, :]
    
    # Concatenate along channel dimension  
    expanded = torch.cat([
        center, up, down, left, right, up_left, up_right, down_left, down_right
    ], dim=-1)  # Shape: (batch, time, height, width, channels * 9)
    
    return expanded

def preprocess_ndws_temporal(x, y):
    """
    Preprocessing that applies spatial encoding to meaningful features per day
    
    Args:
        x: Temporal data (2, height, width, features)
        y: Target fire masks (height, width, 1)
    """
    # Convert to tensors with proper device placement
    if not isinstance(x, torch.Tensor):
        x = torch.FloatTensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.FloatTensor(y)
    
    # Separate Day 0 and Day 1
    day0 = x[0]  # Shape: (height, width, 15)
    day1 = x[1]  # Shape: (height, width, 15) - but only first 10 features are meaningful
    
    # Apply spatial encoding to meaningful features only
    
    # Day 0: Use all 15 features (current weather + terrain + vegetation + human + PrevFireMask)
    day0_expanded = add_surrounding_position(day0.unsqueeze(0).unsqueeze(0))[0, 0]
    # Result: (height, width, 15 * 9) = (height, width, 135)
    
    # Day 1: Only use first 10 meaningful features (forecast weather + static features)
    day1_meaningful = day1[:, :, :Config.day1_input_features]  # (height, width, 10)
    day1_expanded = add_surrounding_position(day1_meaningful.unsqueeze(0).unsqueeze(0))[0, 0]
    # Result: (height, width, 10 * 9) = (height, width, 90)
    
    # Pad Day 1 to match Day 0 dimensions for model compatibility
    day1_padded = F.pad(day1_expanded, (0, 135 - 90), mode='constant', value=0)
    # Result: (height, width, 135) with 45 zero channels at the end
    
    # Recombine
    x_processed = torch.stack([day0_expanded, day1_padded], dim=0)
    
    # Ensure proper target dimensions
    if len(y.shape) == 3:  # Add channel dimension if missing
        y = y.unsqueeze(-1)
   
    return x_processed.to(device), y.to(device)

def preprocess_ndws(x, y):
    """Use the temporal preprocessing function"""
    # Temporarily disable surrounding position encoding for debugging
    if not isinstance(x, torch.Tensor):
        x = torch.FloatTensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.FloatTensor(y)
    
    # Target should already be (N, H, W, 1) from load_ndws_data
    # Don't add extra dimensions
   
    return x.to(device), y.to(device)
    # return preprocess_ndws_temporal(x, y)

def mse_loss(y_true, y_pred):
    """MSE loss for spatial fire prediction"""
    return F.mse_loss(y_pred.squeeze(-1), y_true.squeeze(-1))

def build_model(input_shape=(2, Config.max_height, Config.max_width, Config.max_features_per_day*9),
                embed_dim=Config.embed_dim, num_heads=8, attention_dropout=0.1, dropout=0.2):
   
    model = FlameAIModel(
        input_shape=input_shape, embed_dim=embed_dim, num_heads=num_heads,
        attention_dropout=attention_dropout, dropout=dropout
    ).to(device)
   
    return model

class FlameDataset(Dataset):
    """Dataset class for NDWS 2-day temporal wildfire prediction"""
    def __init__(self, x, y, train=True):
        self.x = x  # Temporal sequences (N, 2, H, W, features)
        self.y = y  # Target fire masks (N, H, W, 1)
        self.train = train
       
    def __len__(self):
        return len(self.x)
   
    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
       
        # Apply augmentation if training (could add spatial augmentations)
        if self.train:
            # Could add spatial augmentations here (rotation, flip, etc.)
            pass
           
        x, y = preprocess_ndws(x, y)
        return x, y

def create_dataloader(x, y, train=True):
    """Create DataLoader for NDWS data"""
    dataset = FlameDataset(x, y, train)
    return DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=train,
        num_workers=0,  # Set to 0 for debugging, increase for performance
        pin_memory=False  # Data is already moved to GPU in preprocess_ndws
    )

def train_model(model, train_loader, epochs, use_custom_loss=True):
    """Training function with custom loss and metrics for NDWS"""
    model.train()
    
    # Use Adam optimizer with constant learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.004)
    
    # Initialize custom loss function
    if use_custom_loss:
        criterion = CustomWBCEDiceLoss(
            w_fire=Config.fire_weight,
            w_no_fire=Config.no_fire_weight, 
            dice_weight=Config.dice_weight
        )
    else:
        criterion = nn.MSELoss()
   
    step = 0
    for epoch in range(epochs):
        total_loss = 0
        total_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}
        num_batches = 0
       
        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                optimizer.zero_grad()
            
                output = model(data)
                    
                if use_custom_loss:
                    loss = criterion(output, target)
                else:
                    loss = mse_loss(target, output)
                
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
                step += 1
            
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                                f'Loss: {loss.item():.6f}, F1: {metrics["f1"]:.4f}, '
                                f'IoU: {metrics["iou"]:.4f}, LR: 0.004')
                            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
       
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_metrics = {key: val / num_batches for key, val in total_metrics.items()}
        
            print(f'Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.6f}, '
                    f'Avg F1: {avg_metrics["f1"]:.4f}, Avg IoU: {avg_metrics["iou"]:.4f}, '
                    f'Avg Precision: {avg_metrics["precision"]:.4f}, Avg Recall: {avg_metrics["recall"]:.4f}')
        else:
            print(f'Epoch {epoch+1}/{epochs} - No valid batches processed!')

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
    
    # Create temporal sequences for 2-day prediction
    print("Creating temporal sequences...")
    temporal_data, temporal_targets = create_temporal_sequence(features, targets)
    print(f"Created temporal data shape: {temporal_data.shape}")
    print(f"Target data shape: {temporal_targets.shape}")
    
    # Single training run (no folds)
    print(f"\n" + "="*50)
    print(f"Starting FLAME AI Training")
    print("="*50)
    
    seed_everything(0)
   
    # Use temporal data for training
    train_x = temporal_data
    train_y = temporal_targets
    
    # Calculate training steps
    train_steps = len(train_x) // Config.batch_size
    train_sample = train_steps * Config.batch_size

    # Trim data to fit batch size
    train_x = train_x[:train_sample]
    train_y = train_y[:train_sample]

    print(f"Training samples: {len(train_x)}")
    print(f"Training steps per epoch: {train_steps}")

    # Create data loader
    train_loader = create_dataloader(train_x, train_y, train=True)
   
    # Build model
    model = build_model(
        input_shape=(2, Config.max_height, Config.max_width, Config.max_features_per_day),  # No 9x expansion
        embed_dim=Config.embed_dim,
        num_heads=8,
        attention_dropout=0.1,
        dropout=0.2
    )
   
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on {len(train_loader)} batches")
   
    # Train model
    train_model(model, train_loader, Config.epochs, use_custom_loss=True)
   
    # Save model
    model_save_path = f'flame_ai_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'embed_dim': Config.embed_dim,
            'max_height': Config.max_height,
            'max_width': Config.max_width,
            'batch_size': Config.batch_size,
            'epochs': Config.epochs,
            'temporal_sequence': Config.temporal_sequence,
            'num_features': Config.max_features_per_day,
            'fire_weight': Config.fire_weight,
            'dice_weight': Config.dice_weight,
        },
        'dataset_info': {
            'num_samples': len(train_x),
            'input_shape': train_x.shape,
            'target_shape': train_y.shape,
            'features': ENHANCED_INPUT_FEATURES,
        }
    }, model_save_path)
   
    print(f"Model saved as {model_save_path}")
   
    # Clear memory
    del model, train_loader
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    print("\n" + "="*60)
    print("FLAME AI Training Complete!")
    print("="*60) 