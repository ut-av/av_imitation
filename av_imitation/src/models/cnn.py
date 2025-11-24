import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels, output_steps, dropout=0.1):
        super(CNN, self).__init__()
        self.output_steps = output_steps
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 120x160 -> 60x80
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 30x40 -> 15x20
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x10 -> 4x5
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_steps * 2)
        )
        
    def forward(self, x):
        # x: (B, C_in, H, W)
        x = self.features(x)
        x = self.head(x)
        # Reshape to (B, T_out, 2)
        x = x.view(-1, self.output_steps, 2)
        return x
