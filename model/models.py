import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return torch.FloatTensor(pos_encoding)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = (1 / self.head_dim ** 0.5)
        self.in_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, hidden_state):
        # Shape: (batch, sequence, height, width, dim)
        batch_size, seq_len, height, width, _ = hidden_state.shape

        # Project inputs to query, key, value
        qkv = self.in_proj(hidden_state)  # Shape: (batch, sequence, height, width, embed_dim * 3)

        # Reshape to (batch, sequence, height, width, num_heads, 3 * head_dim)
        qkv = qkv.view(batch_size, seq_len, height, width, self.num_heads, 3 * self.head_dim)

        # Split into query, key, value
        query, key, value = qkv.chunk(3, dim=-1)  # Shape: (batch, sequence, height, width, num_heads, head_dim)

        # Calculate attention weights across the spatial dimensions and the sequence dimension
        attn_weights = torch.einsum("bshwid,blhwid->bhwisl", query, key) * self.scale

        # Create causal mask
        lower_x = torch.tril(torch.ones((seq_len, seq_len), device=hidden_state.device))
        upper_x = 1 - lower_x
        mask = (lower_x[None, None, None, None, :, :] * 1.0) + (upper_x[None, None, None, None, :, :] * -1000.0)
        attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1)  # Normalize attention weights
        attn_weights = self.dropout(attn_weights)

        # Calculate attention output
        attn_output = torch.einsum("bhwisl,blhwid->bshwid", attn_weights, value)

        # Reshape output to (batch, sequence, height, width, embed_dim)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, height, width, self.embed_dim)

        # Final output projection
        output = self.out_proj(attn_output)

        return self.dropout1(output)

class LateDropout(nn.Module):
    def __init__(self, rate, start_step=0):
        super().__init__()
        self.rate = rate
        self.start_step = start_step
        self.dropout = nn.Dropout(rate)
        self.register_buffer('train_counter', torch.tensor(0, dtype=torch.long))

    def forward(self, inputs):
        if self.training:
            if self.train_counter < self.start_step:
                self.train_counter += 1
                return inputs
            else:
                self.train_counter += 1
                return self.dropout(inputs)
        else:
            return inputs

# Clean, modular blocks inspired by U-Net structure
class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
    
    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    """Downsampling block with max pooling"""
    def __init__(self, in_channels, out_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size)
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class UpSample(nn.Module):
    """Upsampling block with transpose convolution"""
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=kernel_size)
        # After concat: out_channels (upsampled) + skip_channels (skip connection)
        concat_channels = out_channels + skip_channels
        self.conv = ConvBlock(concat_channels, out_channels, dropout=dropout)
    
    def forward(self, x, skip_connection=None):
        x = self.up(x)
        if skip_connection is not None:
            x = torch.cat([skip_connection, x], dim=1)
        x = self.conv(x)
        return x

class SpatialAttention(nn.Module):
    """Simple spatial attention mechanism"""
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)

    def forward(self, inputs):
        # Global average pooling
        nn_out = F.adaptive_avg_pool2d(inputs, 1)  # (batch, channels, 1, 1)
        nn_out = nn_out.squeeze(-1).transpose(-1, -2)  # (batch, 1, channels)
        nn_out = self.conv(nn_out)
        nn_out = torch.sigmoid(nn_out).transpose(-1, -2).unsqueeze(-1)  # (batch, channels, 1, 1)
        return inputs * nn_out

class CNNModel(nn.Module):
    """
    Clean U-Net inspired CNN with Local and Global branches for wildfire prediction
    
    Architecture:
    - Local Branch: Processes features at full resolution for fine details
    - Global Branch: U-Net style encoder-decoder for contextual understanding
    - Feature Fusion: Combines both branches for comprehensive feature extraction
    """
    
    def __init__(self, input_shape, local_eca=False, embed_dim=128, dropout=0.2):
        super().__init__()
        self.input_shape = input_shape
        self.input_channels = input_shape[-1]
        
        # === LOCAL BRANCH (Fine-grained features) ===
        self.local_conv1 = ConvBlock(self.input_channels, 32, dropout=dropout)
        self.local_conv2 = ConvBlock(32, 64, dropout=dropout)  
        self.local_conv3 = ConvBlock(64, 128, dropout=dropout)
        self.local_attention = SpatialAttention(128) if local_eca else nn.Identity()
        
        # === GLOBAL BRANCH (U-Net style encoder-decoder) ===
        # Encoder
        self.enc1 = ConvBlock(self.input_channels, 32, dropout=dropout)
        self.enc2 = DownSample(32, 64, dropout=dropout)
        self.enc3 = DownSample(64, 128, dropout=dropout)  
        self.enc4 = DownSample(128, 192, dropout=dropout)
        
        # Bottleneck
        self.bottleneck = DownSample(192, 256, dropout=dropout)
        
        # Decoder with skip connections
        self.dec4 = UpSample(256, 192, skip_channels=192, dropout=dropout)  # skip from enc4
        self.dec3 = UpSample(192, 128, skip_channels=128, dropout=dropout)  # skip from enc3
        self.dec2 = UpSample(128, 64, skip_channels=64, dropout=dropout)    # skip from enc2
        self.dec1 = UpSample(64, 128, skip_channels=32, dropout=dropout)    # skip from enc1
        
        # === FEATURE FUSION ===
        self.fusion = nn.Sequential(
            ConvBlock(128 + 128, embed_dim, dropout=dropout),  # Local + Global
            ECA(kernel_size=3),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        
    def forward(self, x):
        """
        Forward pass through Local and Global branches
        
        Args:
            x: Input tensor (batch, time_steps, height, width, channels)
            
        Returns:
            features: Feature tensor (batch, time_steps, height, width, embed_dim)
        """
        batch_size, seq_len, height, width, channels = x.shape
        
        # Reshape for conv processing: (batch*seq, channels, height, width)
        x = x.view(batch_size * seq_len, height, width, channels)
        x = x.permute(0, 3, 1, 2)  # Convert to (batch*seq, channels, H, W)
        
        # === LOCAL BRANCH PROCESSING ===
        local = self.local_conv1(x)
        local = self.local_conv2(local) 
        local = self.local_conv3(local)
        local = self.local_attention(local)
        
        # === GLOBAL BRANCH PROCESSING (U-Net) ===
        # Encoder with skip connections
        e1 = self.enc1(x)           # 32 channels
        e2 = self.enc2(e1)          # 64 channels  
        e3 = self.enc3(e2)          # 128 channels
        e4 = self.enc4(e3)          # 192 channels
        
        # Bottleneck
        bottleneck = self.bottleneck(e4)  # 256 channels
        
        # Decoder with skip connections
        d4 = self.dec4(bottleneck, e4)    # 192 channels
        d3 = self.dec3(d4, e3)            # 128 channels  
        d2 = self.dec2(d3, e2)            # 64 channels
        global_features = self.dec1(d2, e1)  # 128 channels
        
        # === FEATURE FUSION ===
        # Combine local and global features
        combined = torch.cat([local, global_features], dim=1)  # 256 channels
        features = self.fusion(combined)  # embed_dim channels
        
        # Convert back to original format: (batch*seq, embed_dim, H, W) -> (batch, seq, H, W, embed_dim)
        features = features.permute(0, 2, 3, 1)  # (batch*seq, H, W, embed_dim)
        features = features.view(batch_size, seq_len, height, width, -1)
        
        return features

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
       
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size, padding=padding, bias=False
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
       
        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
       
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
       
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), return_sequences=True, padding='same'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.return_sequences = return_sequences
       
        # Calculate padding
        if padding == 'same':
            self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        else:
            self.padding = (0, 0)
           
        self.cell = ConvLSTMCell(input_dim, hidden_dim, self.kernel_size, self.padding)

    def forward(self, input_tensor, hidden_state=None):
        b, seq_len, c, h, w = input_tensor.size()
        
        # Validate input dimensions
        assert c == self.input_dim, f"Expected input channels {self.input_dim}, got {c}"
       
        if hidden_state is None:
            h_0 = torch.zeros(b, self.hidden_dim, h, w, device=input_tensor.device, dtype=input_tensor.dtype)
            c_0 = torch.zeros(b, self.hidden_dim, h, w, device=input_tensor.device, dtype=input_tensor.dtype)
        else:
            h_0, c_0 = hidden_state

        outputs = []
        h, c = h_0, c_0
       
        for t in range(seq_len):
            h, c = self.cell(input_tensor[:, t], (h, c))
            if self.return_sequences:
                outputs.append(h)
       
        if self.return_sequences:
            return torch.stack(outputs, dim=1), (h, c)
        else:
            return h, (h, c)

class NextFramePredictor(nn.Module):
    """
    Next Frame Predictor for NDWS 2-day temporal prediction
    Predicts next-day fire mask from current day features and forecast data
    """
    def __init__(self, embed_dim=128, num_convlstm=1):
        super().__init__()
       
        # Single ConvLSTM layer for 2-day temporal modeling
        self.conv_lstm = ConvLSTM(embed_dim, embed_dim, kernel_size=(3, 3), return_sequences=False)
        
        # Final prediction layer
        self.prediction_head = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        """
        Predict next-day fire mask from 2-day temporal sequence
        
        Args:
            inputs: Tensor of shape (batch, 2, height, width, embed_dim)
                   Day 0: Current conditions + PrevFireMask
                   Day 1: Forecast conditions (for temporal context)
        
        Returns:
            fire_prediction: Tensor of shape (batch, height, width, 1)
        """
        # Convert to ConvLSTM format: (batch, seq, channels, height, width)
        x = inputs.permute(0, 1, 4, 2, 3)
        
        # Process through ConvLSTM (2-day sequence -> final state)
        final_state, _ = self.conv_lstm(x)
        
        # Generate fire prediction from final state
        fire_prediction = self.prediction_head(final_state)
        
        # Convert back to (batch, height, width, 1)
        fire_prediction = fire_prediction.permute(0, 2, 3, 1)
        
        return fire_prediction

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_dropout, dropout):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attention_dropout)
        # Keep a projection for non-linearity, but maintain embed_dim dimensions
        self.project = nn.Linear(embed_dim, embed_dim, bias=False)  # 128 → 128 (keep dimensions)
        self.convlstm = ConvLSTM(embed_dim, embed_dim, kernel_size=(3, 3), return_sequences=True)
       
    def forward(self, inputs):
        residual = inputs
        # Apply projection with swish activation (non-linearity preserved!)
        x = F.silu(self.project(inputs))  # swish activation - RESTORED!
        x = self.layer_norm1(x)
        x = self.attn(x)
        x = x + residual
       
        # Convert for ConvLSTM: (batch, seq, height, width, channels) -> (batch, seq, channels, height, width)
        x_conv = x.permute(0, 1, 4, 2, 3)
        x_conv, _ = self.convlstm(x_conv)
        # Convert back: (batch, seq, channels, height, width) -> (batch, seq, height, width, channels)
        x = x_conv.permute(0, 1, 3, 4, 2)
       
        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, attention_dropout, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, attention_dropout, dropout)
            for _ in range(num_layers)
        ])
        self.pos_encoding = nn.Parameter(positional_encoding(150, embed_dim))
       
    def forward(self, inputs):
        x = inputs
        seq_len = x.shape[1]
        pos_embed = self.pos_encoding[:seq_len].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = x + pos_embed.to(x.dtype)
       
        for block in self.blocks:
            x = block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, ld_start_step=55*5):
        super().__init__()
        self.dense = nn.Linear(128, 1, bias=True)
        self.dropout = LateDropout(rate=0.8, start_step=ld_start_step)
        self.layer_norm = nn.LayerNorm(128)
       
    def forward(self, inputs):
        x = inputs
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.dense(x[:, :, :113]))
        return x

class FlameAIModel(nn.Module):
    """
    FLAME AI Model for NDWS 2-day wildfire prediction
    
    Combines spatial CNN feature extraction with temporal modeling for
    next-day wildfire spread prediction using current conditions and forecasts
    """
    def __init__(self, input_shape, embed_dim=128, num_heads=8, attention_dropout=0.1, dropout=0.2):
        super().__init__()
       
        self.cnn = CNNModel(input_shape, local_eca=False, embed_dim=embed_dim, dropout=dropout)
        self.next_frame_predictor = NextFramePredictor(embed_dim=embed_dim)
        
        # Optional: Keep transformer for temporal reasoning
        self.transformer = Transformer(embed_dim, num_heads, 1, attention_dropout, dropout)
        self.use_transformer = False

    def forward(self, x):
        """
        Forward pass for NDWS wildfire prediction
        
        Args:
            x: Input tensor of shape (batch, 2, height, width, channels)
               Day 0: Current conditions + PrevFireMask  
               Day 1: Forecast conditions
               
        Returns:
            fire_prediction: Tensor of shape (batch, height, width, 1)
        """
        # Extract spatial features through CNN
        features = self.cnn(x)  # (batch, 2, height, width, embed_dim)
        
        # Optional transformer processing for temporal reasoning
        if self.use_transformer:
            features = self.transformer(features)
        
        # Predict next-day fire mask
        fire_prediction = self.next_frame_predictor(features)
        
        return fire_prediction

# Custom Loss Function for NDWS
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