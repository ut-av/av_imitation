import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self, input_channels, output_steps, dropout=0.1):
        super(Transformer, self).__init__()
        self.output_steps = output_steps
        
        # input_channels is channels per frame (C)
        self.img_channels = input_channels
        
        # CNN Backbone for feature extraction per frame
        self.backbone = nn.Sequential(
            nn.Conv2d(self.img_channels, 16, kernel_size=5, stride=2, padding=2),
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
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        self.num_frames = T # Store for export if needed
        
        # Process each frame through backbone
        # Merge B and T for batch processing
        x = x.view(B * T, C, H, W)
        features = self.backbone(x) # (B*T, d_model)
        
        # Reshape back to (B, T, d_model)
        features = features.view(B, T, self.d_model)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Transformer
        out = self.transformer_encoder(features)
        
        # Use the last token's output for prediction
        last_token = out[:, -1, :]
        
        pred = self.head(last_token)
        pred = pred.view(B, self.output_steps, 2)
        
        return pred


class TransformerOnnx(nn.Module):
    """ONNX-compatible version of Transformer using global average pooling via mean()."""
    
    def __init__(self, input_channels, output_steps, dropout=0.1):
        super(TransformerOnnx, self).__init__()
        self.output_steps = output_steps
        
        self.img_channels = input_channels
            
        # CNN Backbone for feature extraction per frame (without AdaptiveAvgPool2d)
        self.backbone_conv = nn.Sequential(
            nn.Conv2d(self.img_channels, 16, kernel_size=5, stride=2, padding=2),
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
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Process each frame through backbone
        x = x.view(B * T, C, H, W)
        x = self.backbone_conv(x)
        
        # Global average pooling using mean (ONNX compatible)
        features = x.mean(dim=[2, 3])  # (B*T, d_model)
        
        # Reshape back to (B, T, d_model)
        features = features.view(B, T, self.d_model)
        
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
        input_channels = transformer_model.img_channels
        output_steps = transformer_model.output_steps
        dropout = transformer_model.pos_encoder.dropout.p
        
        onnx_model = cls(input_channels, output_steps, dropout=dropout)
        
        # Copy backbone conv layers (backbone without AdaptiveAvgPool2d and Flatten)
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
