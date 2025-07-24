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
    max_height = 128
    batch_size = 16
    epochs = 20
    total_sequence = 20 + 5
    awp_lambda = 0.04
    num_awp_epoch = 5

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

input_path = 'data/processed'

train_df = pd.read_csv(os.path.join(input_path,'train.csv'))
train_df.head()

# load data
def load_dataX(idx, df):
    csv_file = df.reset_index().to_dict(orient='list')
    dir_path = os.path.join(input_path, "train")
    id = csv_file['id'][idx]
    nt, Nx, Ny = csv_file['Nt'][idx], csv_file['Nx'][idx], csv_file['Ny'][idx]
    theta = np.fromfile(os.path.join(dir_path, csv_file['theta_filename'][idx]), dtype="<f4").reshape(nt, Nx, Ny)
    ustar = np.fromfile(os.path.join(dir_path, csv_file['ustar_filename'][idx]), dtype="<f4").reshape(nt, Nx, Ny)
    xi_f = np.fromfile(os.path.join(dir_path, csv_file['xi_filename'][idx]), dtype="<f4").reshape(nt, Nx, Ny)
    uin  = csv_file['u'][idx]
    alpha = csv_file['alpha'][idx]

    return theta, ustar, xi_f, uin, alpha

def add_surrounding_position(array):
    # Assuming `array` shape is (batch_size, time_steps, height, width, channels)
    batch_size, time_steps, height, width, channels = array.shape

    # Padding the array: pad along the height and width dimensions only
    padded_array = F.pad(array, (0, 0, 1, 1, 1, 1, 0, 0, 0, 0), mode='constant')

    # Extracting the surrounding pixels with appropriate padding
    up = padded_array[:, :, :-2, 1:-1, :]  # 1 pixel up
    down = padded_array[:, :, 2:, 1:-1, :]  # 1 pixel down
    left = padded_array[:, :, 1:-1, :-2, :]  # 1 pixel left
    right = padded_array[:, :, 1:-1, 2:, :]  # 1 pixel right
    up_left = padded_array[:, :, :-2, :-2, :]  # 1 pixel up, 1 pixel left
    up_right = padded_array[:, :, :-2, 2:, :]  # 1 pixel up, 1 pixel right
    down_left = padded_array[:, :, 2:, :-2, :]  # 1 pixel down, 1 pixel left
    down_right = padded_array[:, :, 2:, 2:, :]  # 1 pixel down, 1 pixel right

    # Concatenating the original array with the surrounding pixels
    combined_array = torch.cat([
        array,
        up, down, left, right, up_left, up_right, down_left, down_right
    ], dim=-1)  # Concatenating along the channel dimension

    return combined_array

theta_max = 1000#670
ustar_max = 50#39
uin_max = 50#9
alpha_max = 50#25

theta_min = 240
ustar_min = -4.5
uin_min = 2.0
alpha_min = 0

def normalize(data, min_val, max_val):
    return data / max_val

def load_data_with_next_token(df, input_path, window_size=Config.total_sequence+1):
    spatial_data = []
    labels = []

    for idx in range(len(df)):
        # Load the 3D data (theta, ustar, xi_f)
        theta, ustar, xi_f, uin, alpha = load_dataX(idx, df)
        uin = np.full_like(theta, uin)
        alpha = np.full_like(theta, alpha)
        total_timesteps = theta.shape[0]

        # Generate windows
        for start in range(total_timesteps - window_size):
            # Stack the spatial data for the current window (the same window size for input and label)
            spatial_window = np.stack([
                xi_f[start:start + window_size],   # Use xi_f as is
                normalize(theta[start:start + window_size], theta_min, theta_max), # Normalize theta
                normalize(ustar[start:start + window_size], ustar_min, ustar_max),
                normalize(uin[start:start + window_size], uin_min, uin_max),     # Normalize uin
                normalize(alpha[start:start + window_size], alpha_min, alpha_max),     # Normalize alpha
            ], axis=-1)  # Shape (window_size, Nx, Ny, num_channels)

            # Append the spatial window to the input data
            spatial_data.append(spatial_window)

    # Convert lists to numpy arrays
    spatial_data = np.array(spatial_data)
   
    return spatial_data

spatial_data = load_data_with_next_token(train_df, input_path)

# Check shapes
print("Spatial Data Shape:", spatial_data.shape)  # Should be (num_samples, window_size, Nx, Ny, num_channels)

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

class MLPLayer(nn.Module):
    def __init__(self, embed_dim=128, dropout=0.2):
        super().__init__()
        self.layer_norm = nn.BatchNorm2d(embed_dim, momentum=0.05)  # momentum=1-0.95
        self.fc1 = nn.Linear(embed_dim, embed_dim*2, bias=False)
        self.fc2 = nn.Linear(embed_dim*2, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
       
    def forward(self, inputs):
        # inputs shape: (batch, seq, height, width, embed_dim)
        batch_size, seq_len, height, width, embed_dim = inputs.shape
       
        # Reshape for batch norm: (batch*seq, embed_dim, height, width)
        x = inputs.permute(0, 1, 4, 2, 3).contiguous().view(-1, embed_dim, height, width)
        x = self.layer_norm(x)
        # Reshape back: (batch, seq, height, width, embed_dim)
        x = x.view(batch_size, seq_len, embed_dim, height, width).permute(0, 1, 3, 4, 2)
       
        x = F.silu(self.fc1(x))  # swish activation
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + inputs

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
    def __init__(self, in_channels, channel_size, kernel_size, drop_rate=0.2, expand_ratio=1):
        super().__init__()
        self.fc_expand = nn.Linear(in_channels, channel_size * expand_ratio, bias=False)
        self.bn = nn.BatchNorm2d(channel_size * expand_ratio, momentum=0.05)
        self.fc_project = nn.Linear(channel_size * expand_ratio, channel_size, bias=False)
        self.dwconv = CausalDWConv2D(channel_size * expand_ratio, kernel_size)
        self.dropout = nn.Dropout(drop_rate)
        self.eca = ECA(kernel_size=3)
        self.project_conv = nn.Conv2d(channel_size * expand_ratio, 1, kernel_size=1, padding=0)

    def forward(self, inputs, eca=True):
        # inputs shape: (batch*seq, height, width, in_channels)
        batch_seq, height, width, in_channels = inputs.shape
       
        # Step 1: Expand the feature map
        x = F.silu(self.fc_expand(inputs))  # swish activation
       
        # Convert to conv format: (batch*seq, channels, height, width)
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

        # Convert back to linear format: (batch*seq, height, width, channels)
        x = x.permute(0, 2, 3, 1)
       
        # Step 5: Project back to original channel size
        x = self.fc_project(x)
        x = self.dropout(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, input_shape, local_eca=False, embed_dim=128, dropout=0.2):
        super().__init__()
        self.input_shape = input_shape
       
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

    def forward(self, inputs):
        batch_size, seq_len = inputs.shape[:2]
       
        # Reshape to (batch*seq, height, width, channels)
        reshaped_inputs = inputs.view(-1, *inputs.shape[2:])
       
        # Local Branch (No pooling)
        local_branch = self.local_conv_block_1(reshaped_inputs, eca=self.local_eca)
        local_branch = self.local_conv_block_2(local_branch, eca=self.local_eca)
        local_branch = self.local_conv_block_3(local_branch, eca=self.local_eca)

        # Global Branch (Pooling and keeping track of downsampled layers)
        global_branch = self.global_conv_block_1(reshaped_inputs, eca=True)
        # Convert to conv format for pooling
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
       
        # Reshape back to (batch, seq, height, width, embed_dim)
        features = features.view(batch_size, seq_len, *features.shape[1:])
       
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
        self.kernel_size = kernel_size
        self.return_sequences = return_sequences
       
        # Calculate padding
        if padding == 'same':
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            self.padding = (0, 0)
           
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, self.padding)

    def forward(self, input_tensor, hidden_state=None):
        b, seq_len, c, h, w = input_tensor.size()
       
        if hidden_state is None:
            h_0 = torch.zeros(b, self.hidden_dim, h, w, device=input_tensor.device)
            c_0 = torch.zeros(b, self.hidden_dim, h, w, device=input_tensor.device)
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
    def __init__(self, embed_dim=128, num_convlstm=2):
        super().__init__()
       
        # Create a list of ConvLSTM layers
        self.conv_lstm_layers = nn.ModuleList([
            ConvLSTM(embed_dim, embed_dim, kernel_size=(3, 3), return_sequences=False)
            for _ in range(num_convlstm)
        ])

    def forward(self, inputs, start=0, n_predict=Config.total_sequence, return_states=False, initial_states=None):
        """
        Predicts the next frames given the input frames.
        """
        x = inputs.permute(0, 1, 4, 2, 3)  # Convert to (batch, seq, channels, height, width)
        frames = []  # List to store predicted frames
        batch_size = x.shape[0]

        # Initialize hidden and cell states if not provided
        if initial_states is None:
            h = [x[:, 0] for _ in range(len(self.conv_lstm_layers))]
            c = [x[:, 0] for _ in range(len(self.conv_lstm_layers))]
        else:
            h, c = initial_states

        # Loop to predict n_predict frames
        for i in range(start, n_predict):
            for layer_idx, layer in enumerate(self.conv_lstm_layers):
                if i < 5 and layer_idx == 0:
                    # For the first frame prediction with the first ConvLSTM layer
                    x1, (h[layer_idx], c[layer_idx]) = layer(
                        x[:, i:i+1],
                        (h[layer_idx], c[layer_idx]) if i != 0 else None
                    )
                else:
                    # For subsequent frame predictions
                    x1, (h[layer_idx], c[layer_idx]) = layer(
                        x1.unsqueeze(1),
                        (h[layer_idx], c[layer_idx]) if i != 0 else None
                    )

                # Residual connection for deeper layers
                if layer_idx > 0:
                    x1 += res
                res = x1  # Save the output for the next layer

            frames.append(x1)  # Store the predicted frame

        # Stack all predicted frames together along the time dimension
        x = torch.stack(frames, dim=1)
       
        # Convert back to (batch, seq, height, width, channels)
        x = x.permute(0, 1, 3, 4, 2)

        # Return either just the predicted frames or also the states
        if return_states:
            return x, h, c
        return x

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_dropout, dropout):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(32)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attention_dropout)
        self.project = nn.Linear(embed_dim, 32, bias=False)
        self.convlstm = ConvLSTM(32, embed_dim, kernel_size=(3, 3), return_sequences=True)
       
    def forward(self, inputs):
        residual = inputs
        x = F.silu(self.project(inputs))  # swish activation
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
    def __init__(self, input_shape=(5, Config.max_height, 32, 5), embed_dim=128, num_heads=8,
                 attention_dropout=0.1, dropout=0.2, num_convlstm=2, ld_start_step=55*5):
        super().__init__()
       
        self.cnn = CNNModel(input_shape, local_eca=False, embed_dim=embed_dim, dropout=dropout)
        self.next_frame_predictor = NextFramePredictor(embed_dim=embed_dim, num_convlstm=num_convlstm)
        self.transformer = Transformer(embed_dim, num_heads, num_convlstm, attention_dropout, dropout)
        self.decoder = Decoder(ld_start_step)

    def forward(self, x):
        x = self.cnn(x)
        x = self.next_frame_predictor(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x

def mse_next_20(y_true, y_pred):
    return F.mse_loss(y_pred[:, 4:24], y_true[:, 4:24])

class CosineDecayScheduler:
    def __init__(self, initial_lr, decay_steps, alpha=1e-4):
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.alpha = alpha
       
    def get_lr(self, step):
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.initial_lr * decayed

def build_model(decay_steps, input_shape=(5, Config.max_height, 32, 5),
                embed_dim=Config.embed_dim, num_heads=8, attention_dropout=0.1,
                dropout=0.2, num_convlstm=2, ld_start_step=55*5):
   
    model = FlameAIModel(
        input_shape=input_shape, embed_dim=embed_dim, num_heads=num_heads,
        attention_dropout=attention_dropout, dropout=dropout,
        num_convlstm=num_convlstm, ld_start_step=ld_start_step
    ).to(device)
   
    scheduler = CosineDecayScheduler(1e-3, decay_steps, alpha=1e-4)
    values = [scheduler.get_lr(i) for i in range(decay_steps)]
    plt.plot(values)
    plt.title('Learning Rate Schedule')
    plt.show()
   
    return model, scheduler

def preprocess(x, y):
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
   
    # Create difference feature
    diff_feature = torch.zeros_like(x[:, :, :, :, :1])
    diff_feature[1:] = x[1:, :, :, :, :1] - x[:-1, :, :, :, :1]
   
    x = torch.cat([x, diff_feature], dim=-1)
    x = add_surrounding_position(x.unsqueeze(0))[0]
   
    # Pad height dimension
    pad_height = Config.max_height - 113
    if pad_height > 0:
        x = torch.cat([x, torch.zeros_like(x[:, :, :pad_height])], dim=2)
   
    return x.to(device), y.to(device)

def augment(x, y):
    # Fire location (index 0) - No noise added
    fire_location = x[:, :, :, :, :1]

    # Temperature (index 1) - Adding noise
    temperature = x[:, :, :, :, 1:2]
    if torch.rand(1).item() < 0.2:
        temperature_noise = torch.uniform(-0.066, 0.066, size=(5, 113, 32, 1))
        temperature = torch.clamp(temperature + temperature_noise, 0, 1)

    # Ustar (index 2) - Adding noise
    ustar = x[:, :, :, :, 2:3]
    if torch.rand(1).item() < 0.2:
        ustar_noise = torch.uniform(-0.052, 0.052, size=(5, 113, 32, 1))
        ustar = ustar + ustar_noise

    # Wind speed (index 3) - Adding noise
    wind_speed = x[:, :, :, :, 3:4]
    if torch.rand(1).item() < 0.2:
        wind_speed_noise = torch.uniform(-0.012, 0.012, size=())
        wind_speed = torch.clamp(wind_speed + wind_speed_noise, 0, 1)

    # Slope (index 4) - Adding noise
    slope = x[:, :, :, :, 4:5]
    if torch.rand(1).item() < 0.2:
        slope_noise = torch.uniform(-0.02, 0.02, size=())
        slope = torch.where(slope == 0, slope, torch.clamp(slope + slope_noise, 0, 1))

    # Rebuild the tensor
    x_augmented = torch.cat([fire_location, temperature, ustar, wind_speed, slope], dim=-1)

    return x_augmented, y

class FlameDataset(Dataset):
    def __init__(self, x, y, train=True):
        self.x = x
        self.y = y
        self.train = train
       
    def __len__(self):
        return len(self.x)
   
    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
       
        # Apply augmentation if training
        if self.train:
            # Commented out for now - can be enabled if needed
            # x, y = augment(x, y)
            pass
           
        x, y = preprocess(x, y)
        return x, y

def create_dataloader(x, y, train=True):
    dataset = FlameDataset(x, y, train)
    return DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=train,
        num_workers=0,  # Set to 0 for debugging, increase for performance
        pin_memory=True if torch.cuda.is_available() else False
    )

def train_model(model, train_loader, scheduler, epochs):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
   
    step = 0
    for epoch in range(epochs):
        total_loss = 0
        total_mse_next_20 = 0
        num_batches = 0
       
        for batch_idx, (data, target) in enumerate(train_loader):
            # Update learning rate
            current_lr = scheduler.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
           
            optimizer.zero_grad()
           
            output = model(data)
            loss = F.mse_loss(output, target)
            mse_20 = mse_next_20(target, output)
           
            loss.backward()
            optimizer.step()
           
            total_loss += loss.item()
            total_mse_next_20 += mse_20.item()
            num_batches += 1
            step += 1
           
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}, MSE_next_20: {mse_20.item():.6f}, '
                      f'LR: {current_lr:.6f}')
       
        avg_loss = total_loss / num_batches
        avg_mse_next_20 = total_mse_next_20 / num_batches
       
        print(f'Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.6f}, '
              f'Avg MSE_next_20: {avg_mse_next_20:.6f}')

# Main training loop
if __name__ == "__main__":
    for i in range(10):
        seed_everything(i)
       
        train_x = spatial_data
        train_steps = len(train_x) // Config.batch_size
        train_sample = train_steps * Config.batch_size

        train_y = train_x[:train_sample, 1:, :113, :, 0]
        train_x = train_x[:train_sample, :5]

        train_loader = create_dataloader(train_x, train_y, train=True)
       
        model, scheduler = build_model(
            Config.epochs * train_steps,
            input_shape=(5, Config.max_height, 32, 6*9),
            embed_dim=Config.embed_dim,
            num_heads=8,
            attention_dropout=0.1,
            dropout=0.2,
            num_convlstm=1,
            ld_start_step=train_steps * Config.num_awp_epoch
        )
       
        print(f"Starting training for fold {i}")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        print(f"Training on {len(train_loader)} batches")
       
        train_model(model, train_loader, scheduler, Config.epochs)
       
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold': i,
            'config': {
                'embed_dim': Config.embed_dim,
                'max_height': Config.max_height,
                'batch_size': Config.batch_size,
                'epochs': Config.epochs,
                'total_sequence': Config.total_sequence,
            }
        }, f'trained_model_fold{i}.pth')
       
        print(f"Model saved as trained_model_fold{i}.pth")
       
        # Clear memory
        del model, train_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None