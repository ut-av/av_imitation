import torch
import torch.nn as nn
import torch.nn.functional as F

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
            # AdaptiveAvgPool2d((1,1)) is ONNX compatible
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
        # x: (B, T, C, H, W) or (B, Channels, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B, T * C, H, W)
        
        x = self.features(x)
        x = self.head(x)
        # Reshape to (B, T_out, 2)
        x = x.view(-1, self.output_steps, 2)
        return x


class CNNOnnx(nn.Module):
    """ONNX-compatible version of CNN using global average pooling via mean()."""
    
    def __init__(self, input_channels, output_steps, dropout=0.1):
        super(CNNOnnx, self).__init__()
        self.output_steps = output_steps
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_steps * 2)
        )
        
    def forward(self, x):
        # x: (B, Channels, H, W)
        x = self.conv_layers(x)
        # Global average pooling using mean (ONNX compatible)
        x = x.mean(dim=[2, 3], keepdim=True)
        x = self.head(x)
        x = x.view(-1, self.output_steps, 2)
        return x
    
    @classmethod
    def from_cnn(cls, cnn_model):
        """Create CNNOnnx from a trained CNN model by copying weights."""
        # Infer input_channels from first conv layer
        input_channels = cnn_model.features[0].in_channels
        output_steps = cnn_model.output_steps
        dropout = cnn_model.head[3].p  # Dropout layer
        
        onnx_model = cls(input_channels, output_steps, dropout=dropout)
        
        # Copy conv layers (features without the final AdaptiveAvgPool2d)
        # The original features has AdaptiveAvgPool2d as the last layer
        conv_state = {}
        for name, param in cnn_model.features.state_dict().items():
            # Skip if it's from a layer index >= 28 (AdaptiveAvgPool2d doesn't have params anyway)
            conv_state[name] = param
        onnx_model.conv_layers.load_state_dict(conv_state)
        
        # Copy head weights
        onnx_model.head.load_state_dict(cnn_model.head.state_dict())
        
        return onnx_model
