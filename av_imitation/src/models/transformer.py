import torch
import torch.nn as nn
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
