import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_channels, output_steps, dropout=0.1):
        super(MLP, self).__init__()
        self.output_steps = output_steps
        
        # Downsample input to reduce dimensionality
        self.downsample = nn.AdaptiveAvgPool2d((32, 32))
        
        flat_dim = input_channels * 32 * 32
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_steps * 2)
        )
        
    def forward(self, x):
        # x: (B, C_in, H, W)
        x = self.downsample(x)
        x = self.net(x)
        x = x.view(-1, self.output_steps, 2)
        return x
