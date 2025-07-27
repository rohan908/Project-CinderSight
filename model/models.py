import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class ECA(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)

    def forward(self, inputs):
        # inputs shape: (batch*seq, channels, height, width)
        nn_out = F.adaptive_avg_pool2d(inputs, 1)  # Global average pooling
        nn_out = nn_out.squeeze(-1).transpose(-1, -2)  # (batch*seq, 1, channels)
        nn_out = self.conv(nn_out)
        nn_out = torch.sigmoid(nn_out).transpose(-1, -2).unsqueeze(-1)  # (batch*seq, channels, 1, 1)
        return inputs * nn_out

class CausalDWConv2D(nn.Module):
    def __init__(self, channels, kernel_size=(17, 17), dilation=(1, 1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad_h = dilation[0] * (kernel_size[0] - 1)
        self.pad_w = dilation[1] * (kernel_size[1] - 1)
       
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size,
            stride=1, dilation=dilation,
            padding=0, bias=False, groups=channels
        )

    def forward(self, inputs):
        # Causal padding: pad left and top, no padding right and bottom
        x = F.pad(inputs, (self.pad_w, 0, self.pad_h, 0), mode='constant', value=0)
        x = self.dw_conv(x)
        return x

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop_rate=0.2):
        super().__init__()
        self.fc_expand = nn.Linear(in_channels, out_channels, bias=False)
        self.dwconv = CausalDWConv2D(out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.05)
        self.project_conv = nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)
        self.eca = ECA(kernel_size=3)
        self.final_fc = nn.Linear(out_channels, out_channels, bias=False)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, inputs, eca=True):
        # inputs shape: (batch, height, width, in_channels)
       
        # Step 1: Expand the feature map
        x = F.silu(self.fc_expand(inputs))  # swish activation
       
        # Convert to conv format: (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
       
        # Step 2: Apply depthwise convolution
        x = self.dwconv(x)
        x = self.bn(x)

        # Step 3: Project to 1 channel for attention
        projected = torch.sigmoid(self.project_conv(x))

        # Step 4: Multiply the depthwise convolution output by projected values for spatial attention
        x = x * projected

        if eca:
            x = self.eca(x)

        # Convert back to linear format: (batch, height, width, channels)
        x = x.permute(0, 2, 3, 1)
       
        # Step 5: Final fully connected layer with dropout
        x = self.final_fc(x)
        x = self.dropout(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, input_shape, local_eca=False, embed_dim=128, dropout=0.2):
        super().__init__()
        # Local Branch blocks
        self.local_conv_block_1 = Conv2DBlock(input_shape[-1], 32, kernel_size=(3, 3), drop_rate=dropout)
        self.local_conv_block_2 = Conv2DBlock(32, 64, kernel_size=(3, 3), drop_rate=dropout)
        self.local_conv_block_3 = Conv2DBlock(64, 128, kernel_size=(3, 3), drop_rate=dropout)
       
        # Global Branch blocks
        self.global_conv_block_1 = Conv2DBlock(input_shape[-1], 32, kernel_size=(3, 3), drop_rate=dropout)
        self.global_maxpool_1 = nn.MaxPool2d(kernel_size=(2, 2))
       
        self.global_conv_block_2 = Conv2DBlock(32, 64, kernel_size=(3, 3), drop_rate=dropout)
        self.global_maxpool_2 = nn.MaxPool2d(kernel_size=(2, 2))
       
        self.global_conv_block_3 = Conv2DBlock(64, 128, kernel_size=(3, 3), drop_rate=dropout)
        self.global_maxpool_3 = nn.MaxPool2d(kernel_size=(2, 2))
       
        self.global_conv_block_4 = Conv2DBlock(128, 192, kernel_size=(3, 3), drop_rate=dropout)
        self.global_maxpool_4 = nn.MaxPool2d(kernel_size=(2, 1))
       
        # Bottleneck
        self.global_bottleneck_block = Conv2DBlock(192, 256, kernel_size=(3, 3), drop_rate=dropout)
       
        # Upsampling layers
        self.global_upsample_1 = nn.ConvTranspose2d(256, 192, kernel_size=(2, 1), stride=(2, 1))
        self.global_conv_block_5 = Conv2DBlock(192 + 192, 192, kernel_size=(3, 3), drop_rate=dropout)
       
        self.global_upsample_2 = nn.ConvTranspose2d(192, 128, kernel_size=(2, 2), stride=(2, 2))
        self.global_conv_block_6 = Conv2DBlock(128 + 128, 128, kernel_size=(3, 3), drop_rate=dropout)
       
        self.global_upsample_3 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
        self.global_conv_block_7 = Conv2DBlock(64 + 64, 64, kernel_size=(3, 3), drop_rate=dropout)
       
        self.global_upsample_4 = nn.ConvTranspose2d(64, 128, kernel_size=(2, 2), stride=(2, 2))
       
        # Final feature block
        self.final_conv_block = Conv2DBlock(128 + 128, embed_dim, kernel_size=(1, 1), drop_rate=dropout)
       
        self.local_eca = local_eca

    def forward(self, x):
        # Input shape: (batch, height, width, channels)
       
        # Local Branch Processing
        local_branch = self.local_conv_block_1(x, eca=self.local_eca)
        local_branch = self.local_conv_block_2(local_branch, eca=self.local_eca)
        local_branch = self.local_conv_block_3(local_branch, eca=self.local_eca)
       
        # Global Branch Processing
        global_branch = self.global_conv_block_1(x)
        down64 = global_branch  # Save this for merging later
        global_branch_conv = global_branch.permute(0, 3, 1, 2)
        global_branch_conv = self.global_maxpool_1(global_branch_conv)
        global_branch = global_branch_conv.permute(0, 2, 3, 1)

        global_branch = self.global_conv_block_2(global_branch)
        down128 = global_branch  # Save this for merging later
        global_branch_conv = global_branch.permute(0, 3, 1, 2)
        global_branch_conv = self.global_maxpool_2(global_branch_conv)
        global_branch = global_branch_conv.permute(0, 2, 3, 1)

        global_branch = self.global_conv_block_3(global_branch)
        down256 = global_branch  # Save this for merging later
        global_branch_conv = global_branch.permute(0, 3, 1, 2)
        global_branch_conv = self.global_maxpool_3(global_branch_conv)
        global_branch = global_branch_conv.permute(0, 2, 3, 1)

        global_branch = self.global_conv_block_4(global_branch)
        down512 = global_branch  # Save this for merging later
        global_branch_conv = global_branch.permute(0, 3, 1, 2)
        global_branch_conv = self.global_maxpool_4(global_branch_conv)
        global_branch = global_branch_conv.permute(0, 2, 3, 1)

        # Bottleneck Layer
        global_branch = self.global_bottleneck_block(global_branch)

        # Global Branch Up-sampling
        global_branch_conv = global_branch.permute(0, 3, 1, 2)
        global_branch_conv = self.global_upsample_1(global_branch_conv)
        global_branch = global_branch_conv.permute(0, 2, 3, 1)
        global_branch = torch.cat([global_branch, down512], dim=-1)  # Merge with down512
        global_branch = self.global_conv_block_5(global_branch)

        global_branch_conv = global_branch.permute(0, 3, 1, 2)
        global_branch_conv = self.global_upsample_2(global_branch_conv)
        global_branch = global_branch_conv.permute(0, 2, 3, 1)
        global_branch = torch.cat([global_branch, down256], dim=-1)  # Merge with down256
        global_branch = self.global_conv_block_6(global_branch)

        global_branch_conv = global_branch.permute(0, 3, 1, 2)
        global_branch_conv = self.global_upsample_3(global_branch_conv)
        global_branch = global_branch_conv.permute(0, 2, 3, 1)
        global_branch = torch.cat([global_branch, down128], dim=-1)  # Merge with down128
        global_branch = self.global_conv_block_7(global_branch)

        global_branch_conv = global_branch.permute(0, 3, 1, 2)
        global_branch_conv = self.global_upsample_4(global_branch_conv)
        global_branch = global_branch_conv.permute(0, 2, 3, 1)

        features = torch.cat([local_branch, global_branch], dim=-1)
        features = self.final_conv_block(features, eca=True)
       
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
    Next Frame Predictor for NDWS fire spread prediction
    Predicts next-day fire mask from current day features and forecast data
    """
    def __init__(self, embed_dim=128):
        super().__init__()

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
        Predict next-day fire mask from feature representation
        
        Args:
            inputs: Tensor of shape (batch, height, width, embed_dim)
        
        Returns:
            fire_prediction: Tensor of shape (batch, height, width, 1)
        """
        
        # Reshape for conv
        x = inputs.permute(0, 3, 1, 2) # (B, C, H, W)

        # Generate fire prediction from final state
        x = self.prediction_head(x)

        # Reshape to output shape
        x = x.permute(0, 2, 3, 1) # (B, H, W, 1)
        
        return x

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
    def __init__(self, ld_start_step=250):
        super().__init__()
        self.dense = nn.Linear(128, 1, bias=True)
        self.dropout = LateDropout(rate=0.8, start_step=ld_start_step)
        self.layer_norm = nn.LayerNorm(128)
       
    def forward(self, inputs):
        x = self.layer_norm(inputs)
        x = self.dropout(x)
        x = torch.sigmoid(self.dense(x))
        return x

class FlameAIModel(nn.Module):
    """
    FLAME AI Model modified for NDWS wildfire prediction
    """
    def __init__(self, input_shape, embed_dim=128, num_heads=8, attention_dropout=0.1, dropout=0.2):
        super().__init__()
       
        self.cnn = CNNModel(input_shape, local_eca=False, embed_dim=embed_dim, dropout=dropout)
        self.next_frame_predictor = NextFramePredictor()

    def forward(self, x):
        """
        Forward pass for NDWS wildfire prediction
        
        Args:
            x: Input tensor of shape (batch, height, width, channels)
               
        Returns:
            fire_prediction: Tensor of shape (batch, height, width, 1)
        """
        # Extract spatial features through CNN
        features = self.cnn(x)  # (batch, height, width, embed_dim)
        
        # Predict next-day fire mask
        fire_prediction = self.next_frame_predictor(features)
        
        return fire_prediction