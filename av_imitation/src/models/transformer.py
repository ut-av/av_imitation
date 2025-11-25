import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self, input_channels, output_steps, dropout=0.1):
        super(Transformer, self).__init__()
        self.output_steps = output_steps
        
        # Assume input_channels is (num_frames * 3)
        # We need to infer num_frames. 
        # But wait, we can't infer it easily if we don't know it's RGB.
        # Let's assume RGB (3 channels per frame).
        self.img_channels = 3
        if input_channels % 3 != 0:
            raise ValueError(f"Input channels {input_channels} not divisible by 3. Assumption of RGB frames failed.")
            
        self.num_frames = input_channels // 3
        
        # CNN Backbone for feature extraction per frame
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # AdaptiveAvgPool2d((1,1)) is ONNX compatible
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.d_model = 64
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        
        # Output Head
        self.head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, output_steps * 2)
        )
        
    def forward(self, x):
        # x: (B, C_in, H, W) -> (B, T, 3, H, W)
        B, C, H, W = x.shape
        x = x.view(B, self.num_frames, 3, H, W)
        
        # Process each frame through backbone
        # Merge B and T for batch processing
        x = x.view(B * self.num_frames, 3, H, W)
        features = self.backbone(x) # (B*T, d_model)
        
        # Reshape back to (B, T, d_model)
        features = features.view(B, self.num_frames, self.d_model)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Transformer
        out = self.transformer_encoder(features)
        
        # Use the last token's output for prediction
        # Or average pool? Let's use last token as it has seen all history.
        last_token = out[:, -1, :]
        
        pred = self.head(last_token)
        pred = pred.view(B, self.output_steps, 2)
        
        return pred


class TransformerOnnx(nn.Module):
    """ONNX-compatible version of Transformer using global average pooling via mean()."""
    
    def __init__(self, input_channels, output_steps, dropout=0.1):
        super(TransformerOnnx, self).__init__()
        self.output_steps = output_steps
        
        self.img_channels = 3
        if input_channels % 3 != 0:
            raise ValueError(f"Input channels {input_channels} not divisible by 3. Assumption of RGB frames failed.")
            
        self.num_frames = input_channels // 3
        
        # CNN Backbone for feature extraction per frame (without AdaptiveAvgPool2d)
        self.backbone_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.d_model = 64
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        
        # Output Head
        self.head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, output_steps * 2)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.num_frames, 3, H, W)
        
        # Process each frame through backbone
        x = x.view(B * self.num_frames, 3, H, W)
        x = self.backbone_conv(x)
        
        # Global average pooling using mean (ONNX compatible)
        features = x.mean(dim=[2, 3])  # (B*T, d_model)
        
        # Reshape back to (B, T, d_model)
        features = features.view(B, self.num_frames, self.d_model)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Transformer
        out = self.transformer_encoder(features)
        
        # Use the last token's output
        last_token = out[:, -1, :]
        
        pred = self.head(last_token)
        pred = pred.view(B, self.output_steps, 2)
        
        return pred
    
    @classmethod
    def from_transformer(cls, transformer_model):
        """Create TransformerOnnx from a trained Transformer model by copying weights."""
        input_channels = transformer_model.num_frames * 3
        output_steps = transformer_model.output_steps
        dropout = transformer_model.pos_encoder.dropout.p
        
        onnx_model = cls(input_channels, output_steps, dropout=dropout)
        
        # Copy backbone conv layers (backbone without AdaptiveAvgPool2d and Flatten)
        # Original backbone: Conv, ReLU, MaxPool, Conv, ReLU, MaxPool, Conv, ReLU, AdaptiveAvgPool2d, Flatten
        # We need layers 0-7 (indices)
        backbone_state = {}
        for name, param in transformer_model.backbone.state_dict().items():
            backbone_state[name] = param
        onnx_model.backbone_conv.load_state_dict(backbone_state)
        
        # Copy transformer encoder
        onnx_model.transformer_encoder.load_state_dict(transformer_model.transformer_encoder.state_dict())
        
        # Copy positional encoding
        onnx_model.pos_encoder.load_state_dict(transformer_model.pos_encoder.state_dict())
        
        # Copy head weights
        onnx_model.head.load_state_dict(transformer_model.head.state_dict())
        
        return onnx_model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
