import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_channels, output_steps, dropout=0.1):
        super(MLP, self).__init__()
        self.output_steps = output_steps
        self.target_size = (32, 32)
        
        # Downsample input to reduce dimensionality
        self.downsample = nn.AdaptiveAvgPool2d(self.target_size)
        
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


class MLPOnnx(nn.Module):
    """ONNX-compatible version of MLP that uses interpolate instead of AdaptiveAvgPool2d."""
    
    def __init__(self, input_channels, output_steps, dropout=0.1):
        super(MLPOnnx, self).__init__()
        self.output_steps = output_steps
        self.target_size = (32, 32)
        
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
        # Use interpolate for ONNX export compatibility
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        x = self.net(x)
        x = x.view(-1, self.output_steps, 2)
        return x
    
    @classmethod
    def from_mlp(cls, mlp_model):
        """Create MLPOnnx from a trained MLP model by copying weights."""
        onnx_model = cls(
            input_channels=mlp_model.net[1].in_features // (32 * 32),
            output_steps=mlp_model.output_steps,
        )
        # Copy the net weights (skip downsample since MLPOnnx doesn't have it)
        onnx_model.net.load_state_dict(mlp_model.net.state_dict())
        return onnx_model
