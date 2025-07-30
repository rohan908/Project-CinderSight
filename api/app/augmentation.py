import random
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF

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
    Args:
        features: torch.Tensor of shape (H, W, C)

    Returns:
        torch.Tensor of shape (H, W, C*9)
    """
    # 1) move channels to dim=0
    x = features.permute(2, 0, 1)          # now shape = (C, H, W)
    # 2) pad H and W by 1 on both sides: pad = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
    p = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)  # shape = (C, H+2, W+2)
    
    C, Hp, Wp = p.shape
    H, W = Hp - 2, Wp - 2

    # 3) extract each neighbor
    center   = p[:, 1:1+H,   1:1+W   ]
    up       = p[:, 0:  H,   1:1+W   ]
    down     = p[:, 2:2+H,   1:1+W   ]
    left     = p[:, 1:1+H,   0:  W   ]
    right    = p[:, 1:1+H,   2:2+W   ]
    up_left  = p[:, 0:  H,   0:  W   ]
    up_right = p[:, 0:  H,   2:2+W   ]
    dn_left  = p[:, 2:2+H,   0:  W   ]
    dn_right = p[:, 2:2+H,   2:2+W   ]

    # 4) concatenate on channel dim
    out = torch.cat([
       center, up, down, left, right,
       up_left, up_right, dn_left, dn_right
    ], dim=0)  # shape = (C*9, H, W)

    # 5) move channels back to last dim
    return out.permute(1, 2, 0)              # (H, W, C*9)