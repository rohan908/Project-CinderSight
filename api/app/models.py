import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
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
        batch_size, seq_len, height, width, _ = hidden_state.shape
        qkv = self.in_proj(hidden_state)
        qkv = qkv.view(batch_size, seq_len, height, width, self.num_heads, 3 * self.head_dim)
        query, key, value = qkv.chunk(3, dim=-1)
        attn_weights = torch.einsum("bshwid,blhwid->bhwisl", query, key) * self.scale
        lower_x = torch.tril(torch.ones((seq_len, seq_len), device=hidden_state.device))
        upper_x = 1 - lower_x
        mask = (lower_x[None, None, None, None, :, :] * 1.0) + (upper_x[None, None, None, None, :, :] * -1000.0)
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.einsum("bhwisl,blhwid->bshwid", attn_weights, value)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, height, width, self.embed_dim)
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
        nn_out = F.adaptive_avg_pool2d(inputs, 1)
        nn_out = nn_out.squeeze(-1).transpose(-1, -2)
        nn_out = self.conv(nn_out)
        nn_out = torch.sigmoid(nn_out).transpose(-1, -2).unsqueeze(-1)
        return inputs * nn_out

class CausalDWConv2D(nn.Module):
    def __init__(self, channels, kernel_size=(17, 17), dilation=(1, 1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = ((kernel_size[0] - 1) * dilation[0], (kernel_size[1] - 1) * dilation[1])
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                             dilation=dilation, padding=self.padding, groups=channels, bias=False)

    def forward(self, inputs):
        x = self.conv(inputs)
        # Apply causal masking
        x = x[:, :, :inputs.shape[2], :inputs.shape[3]]
        return x

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop_rate=0.2):
        super().__init__()
        self.fc_expand = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.dwconv = CausalDWConv2D(out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.eca = ECA()
        self.final_fc = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.project_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, inputs, eca=True):
        residual = inputs
        x = self.fc_expand(inputs)
        x = self.dwconv(x)
        x = self.bn(x)
        x = F.silu(x)
        if eca:
            x = self.eca(x)
        x = self.final_fc(x)
        x = self.dropout(x)
        x = x + self.project_conv(residual)
        return x

class CNNModel(nn.Module):
    def __init__(self, input_shape, local_eca=False, embed_dim=128, dropout=0.2):
        super().__init__()
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        
        # Local Branch
        self.local_conv_block_1 = Conv2DBlock(input_shape[2], embed_dim//2, (17, 17), dropout)
        self.local_conv_block_2 = Conv2DBlock(embed_dim//2, embed_dim//2, (17, 17), dropout)
        self.local_conv_block_3 = Conv2DBlock(embed_dim//2, embed_dim//2, (17, 17), dropout)
        
        # Global Branch
        self.global_conv_block_1 = Conv2DBlock(input_shape[2], embed_dim//2, (17, 17), dropout)
        self.global_conv_block_2 = Conv2DBlock(embed_dim//2, embed_dim//2, (17, 17), dropout)
        self.global_conv_block_3 = Conv2DBlock(embed_dim//2, embed_dim//2, (17, 17), dropout)
        self.global_conv_block_4 = Conv2DBlock(embed_dim//2, embed_dim//2, (17, 17), dropout)
        self.global_bottleneck_block = Conv2DBlock(embed_dim//2, embed_dim//2, (17, 17), dropout)
        
        # Upsampling blocks
        self.global_upsample_1 = nn.ConvTranspose2d(embed_dim//2, embed_dim//2, kernel_size=2, stride=2)
        self.global_conv_block_5 = Conv2DBlock(embed_dim//2, embed_dim//2, (17, 17), dropout)
        self.global_upsample_2 = nn.ConvTranspose2d(embed_dim//2, embed_dim//2, kernel_size=2, stride=2)
        self.global_conv_block_6 = Conv2DBlock(embed_dim//2, embed_dim//2, (17, 17), dropout)
        self.global_upsample_3 = nn.ConvTranspose2d(embed_dim//2, embed_dim//2, kernel_size=2, stride=2)
        self.global_conv_block_7 = Conv2DBlock(embed_dim//2, embed_dim//2, (17, 17), dropout)
        self.global_upsample_4 = nn.ConvTranspose2d(embed_dim//2, embed_dim//2, kernel_size=2, stride=2)
        self.final_conv_block = Conv2DBlock(embed_dim//2, embed_dim//2, (17, 17), dropout)

    def forward(self, x):
        # Input shape: (batch, height, width, channels)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width)
        
        # Local Branch Processing
        local_x = self.local_conv_block_1(x)
        local_x = self.local_conv_block_2(local_x)
        local_x = self.local_conv_block_3(local_x)
        
        # Global Branch Processing
        global_x = self.global_conv_block_1(x)
        global_x = self.global_conv_block_2(global_x)
        global_x = self.global_conv_block_3(global_x)
        global_x = self.global_conv_block_4(global_x)
        global_x = self.global_bottleneck_block(global_x)
        
        # Upsampling path
        global_x = self.global_upsample_1(global_x)
        global_x = self.global_conv_block_5(global_x)
        global_x = self.global_upsample_2(global_x)
        global_x = self.global_conv_block_6(global_x)
        global_x = self.global_upsample_3(global_x)
        global_x = self.global_conv_block_7(global_x)
        global_x = self.global_upsample_4(global_x)
        global_x = self.final_conv_block(global_x)
        
        # Combine local and global features
        combined = torch.cat([local_x, global_x], dim=1)  # (batch, embed_dim, height, width)
        
        # Convert back to (batch, height, width, channels)
        combined = combined.permute(0, 2, 3, 1)
        
        return combined

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=True)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.hidden_dim, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)
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
        self.padding = kernel_size[0] // 2 if padding == 'same' else 0
        
        self.cell = ConvLSTMCell(self.input_dim, self.hidden_dim, self.kernel_size, self.padding)

    def forward(self, input_tensor, hidden_state=None):
        batch_size, seq_len, channels, height, width = input_tensor.size()
        
        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_dim, height, width).to(input_tensor.device)
            c = torch.zeros(batch_size, self.hidden_dim, height, width).to(input_tensor.device)
            hidden_state = (h, c)
        
        output = []
        h, c = hidden_state
        
        for t in range(seq_len):
            h, c = self.cell(input_tensor[:, t, :, :, :], (h, c))
            if self.return_sequences:
                output.append(h)
        
        if self.return_sequences:
            output = torch.stack(output, dim=1)
            return output, (h, c)
        else:
            return h, (h, c)

class NextFramePredictor(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.prediction_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2, embed_dim//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//4, 1, kernel_size=1)
        )

    def forward(self, inputs):
        # Input shape: (batch, height, width, channels)
        x = inputs.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.prediction_head(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, 1)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_dropout, dropout):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attention_dropout)
        self.project = nn.Linear(embed_dim, embed_dim, bias=False)
        self.convlstm = ConvLSTM(embed_dim, embed_dim, kernel_size=(3, 3), return_sequences=True)
       
    def forward(self, inputs):
        residual = inputs
        x = F.silu(self.project(inputs))
        x = self.layer_norm1(x)
        x = self.attn(x)
        x = x + residual
        x_conv = x.permute(0, 1, 4, 2, 3)
        x_conv, _ = self.convlstm(x_conv)
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
    """Complete FlameAI Model for NDWS wildfire prediction"""
    
    def __init__(self, input_shape, embed_dim=128, num_heads=8, attention_dropout=0.1, dropout=0.2, use_decoder=False):
        super().__init__()
        self.cnn = CNNModel(input_shape, local_eca=False, embed_dim=embed_dim, dropout=dropout)
        self.next_frame_predictor = NextFramePredictor()
        self.decoder = Decoder()
        self.use_decoder = use_decoder

    def forward(self, x):
        # Extract spatial features through CNN
        features = self.cnn(x)  # (batch, height, width, embed_dim)
        # Predict next-day fire mask
        prediction = self.decoder(features) if self.use_decoder else self.next_frame_predictor(features)
        return prediction