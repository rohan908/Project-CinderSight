import torch
import torch.nn as nn
import torch.nn.functional as F

class FlameAIModel(nn.Module):
    """Simplified FlameAI Model for visualization purposes"""
    
    def __init__(self, input_shape=(32, 32, 171), embed_dim=128, num_heads=8, 
                 attention_dropout=0.1, dropout=0.2):
        super(FlameAIModel, self).__init__()
        
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Simple CNN backbone for feature extraction
        self.conv1 = nn.Conv2d(input_shape[2], 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, embed_dim, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(embed_dim, 1, kernel_size=1)
        
    def forward(self, x):
        # Input shape: (batch, height, width, channels)
        # Convert to (batch, channels, height, width)
        if x.dim() == 4 and x.shape[1] != self.input_shape[2]:
            x = x.permute(0, 3, 1, 2)
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Final prediction
        x = self.final_conv(x)
        
        # Convert back to (batch, height, width, channels)
        x = x.permute(0, 2, 3, 1)
        
        return torch.sigmoid(x)