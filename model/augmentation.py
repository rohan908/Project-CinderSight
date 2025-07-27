import random
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from train import Config, device

def create_random_crop_params(original_size, crop_size):
    """Create random crop parameters"""
    if crop_size >= original_size:
        return 0, 0
    
    max_offset = original_size - crop_size

    x_offset = random.randint(0, max_offset)
    y_offset = random.randint(0, max_offset)

    return x_offset, y_offset

def apply_random_crop(features, target, crop_size):
    """
    Apply random crop to features and target
    
    Args:
        data: (H, W, features) - input features
        target: (H, W, 1) - target fire mask
        crop_size: int - size of crop
    
    Returns:
        cropped_data: (crop_size, crop_size, features)
        cropped_target: (crop_size, crop_size, 1)
    """
    if crop_size >= features.shape[1]:  # No cropping needed
        return features, target
        
    # Get random crop parameters
    x_offset, y_offset = create_random_crop_params(features.shape[1], crop_size)
    
    # Apply crop to features
    cropped_features = features[x_offset:x_offset+crop_size, 
                                y_offset:y_offset+crop_size, 
                                :]
    
    # Apply same crop to target
    cropped_target = target[x_offset:x_offset+crop_size, 
                            y_offset:y_offset+crop_size, 
                            :]
    
    return cropped_features, cropped_target

def apply_rotation(features, target, angle):
    """
    Apply rotation to features and target
    
    Args:
        features: (H, W, features) - input features
        target: (H, W, 1) - target fire mask
        angle: int - rotation angle (0, 90, 180, 270)
    
    Returns:
        rotated_features: (H, W, features)
        rotated_target: (H, W, 1)
    """
    if angle == 0:
        return features, target
    
    # Convert to torch tensors
    features_tensor = torch.FloatTensor(features)
    target_tensor = torch.FloatTensor(target)
    
    # Rotate features (H, W, features) -> (features, H, W) -> rotate -> (H, W, features)
    features_for_rotation = features_tensor.permute(2, 0, 1)  # (features, H, W)
    rotated_features = TF.rotate(features_for_rotation, angle)
    rotated_features = rotated_features.permute(1, 2, 0).numpy() # (H, W, features)
    
    # Rotate target (H, W, 1) -> (1, H, W) -> rotate -> (H, W, 1)
    target_for_rotation = target_tensor.permute(2, 0, 1)  # (1, H, W)
    rotated_target = TF.rotate(target_for_rotation, angle)
    rotated_target = rotated_target.permute(1, 2, 0).numpy()  # (H, W, 1)
    
    return rotated_features, rotated_target

def check_valid_sample(target, crop_size, min_valid_ratio=0.8):
    """
    Check if a sample has enough valid (non -1) pixels after cropping
    
    Args:
        target: Target fire mask
        crop_size: Size of crop
        min_valid_ratio: Minimum ratio of valid pixels required
    
    Returns:
        bool: True if sample is valid
    """
    total_pixels = crop_size * crop_size
    valid_pixels = np.sum(target != -1.0)
    valid_ratio = valid_pixels / total_pixels
    
    return valid_ratio >= min_valid_ratio

def add_surrounding_position(features):
    """
    Add surrounding position encoding for spatial context
    
    Args:
        features: Shape (B, H, W, features)
    
    Returns:
        expanded: Shape (B, H, W, expanded_features)
    """
    # Padding the array: pad along the height and width dimensions only
    padded_array = F.pad(features, (0, 0, 1, 1, 1, 1, 0, 0, 0, 0), mode='constant')

    # Extract surrounding pixels
    center = features
    up = padded_array[:, :-2, 1:-1, :]
    down = padded_array[:, 2:, 1:-1, :]
    left = padded_array[:, 1:-1, :-2, :]
    right = padded_array[:, 1:-1, 2:, :]
    up_left = padded_array[:, :-2, :-2, :]
    up_right = padded_array[:, :-2, 2:, :]
    down_left = padded_array[:, 2:, :-2, :]
    down_right = padded_array[:, 2:, 2:, :]
    
    # Concatenate along channel dimension  
    expanded = torch.cat([
        center, up, down, left, right, up_left, up_right, down_left, down_right
    ], dim=-1)  # Shape: (B, H, W, channels * 9)
    
    return expanded

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