"""
3D UNet Encoder for IRIS Framework

This module implements a 3D UNet encoder with:
- 4 downsampling stages with doubling channels: [32, 64, 128, 256, 512]
- Residual blocks at each stage
- Skip connections for decoder
- Instance normalization throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """
    3D Residual block with instance normalization.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for the first convolution (default: 1)
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class EncoderStage3D(nn.Module):
    """
    Single encoder stage with multiple residual blocks.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_blocks (int): Number of residual blocks (default: 2)
        downsample (bool): Whether to downsample (default: True)
    """
    
    def __init__(self, in_channels, out_channels, num_blocks=2, downsample=True):
        super().__init__()
        
        self.downsample = downsample
        
        # First block handles channel change and potential downsampling
        stride = 2 if downsample else 1
        self.blocks = nn.ModuleList([
            ResidualBlock3D(in_channels, out_channels, stride=stride)
        ])
        
        # Additional blocks maintain the same channels
        for _ in range(num_blocks - 1):
            self.blocks.append(
                ResidualBlock3D(out_channels, out_channels, stride=1)
            )
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Encoder3D(nn.Module):
    """
    3D UNet Encoder with 4 downsampling stages.
    
    The encoder follows the paper's specification:
    - 4 stages with doubling channels: [32, 64, 128, 256, 512]
    - Residual blocks at each stage
    - Skip connections for decoder
    - Instance normalization throughout
    
    Args:
        in_channels (int): Number of input channels (default: 1 for medical images)
        base_channels (int): Base number of channels (default: 32)
        num_blocks_per_stage (int): Number of residual blocks per stage (default: 2)
    """
    
    def __init__(self, in_channels=1, base_channels=32, num_blocks_per_stage=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder stages
        channels = [base_channels, base_channels * 2, base_channels * 4, 
                   base_channels * 8, base_channels * 16]
        
        self.stages = nn.ModuleList()
        
        # Stage 0: No downsampling, just process initial features
        self.stages.append(
            EncoderStage3D(channels[0], channels[0], num_blocks_per_stage, downsample=False)
        )
        
        # Stages 1-4: Downsampling stages
        for i in range(1, 5):
            self.stages.append(
                EncoderStage3D(channels[i-1], channels[i], num_blocks_per_stage, downsample=True)
            )
        
        # Store channel dimensions for decoder
        self.feature_channels = channels
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.InstanceNorm3d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of the encoder.
        
        Args:
            x: Input tensor (B, C, D, H, W)
        
        Returns:
            features: List of feature maps at different scales
                     [stage0, stage1, stage2, stage3, stage4]
                     Spatial dimensions: [D×H×W, D/2×H/2×W/2, ..., D/16×H/16×W/16]
                     Channels: [32, 32, 64, 128, 256, 512]
        """
        features = []
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Process through encoder stages
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
        
        return features
    
    def get_feature_channels(self):
        """Return the number of channels at each stage."""
        return self.feature_channels


def test_encoder_3d():
    """Test function to verify 3D UNet Encoder works correctly."""
    print("Testing 3D UNet Encoder...")
    
    # Test parameters
    batch_size = 2
    in_channels = 1
    base_channels = 32
    depth, height, width = 64, 128, 128
    
    # Create test input
    x = torch.randn(batch_size, in_channels, depth, height, width)
    print(f"Input shape: {x.shape}")
    
    # Create encoder
    encoder = Encoder3D(
        in_channels=in_channels,
        base_channels=base_channels,
        num_blocks_per_stage=2
    )
    
    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {param_count:,}")
    
    # Forward pass
    with torch.no_grad():
        features = encoder(x)
    
    print(f"Number of feature maps: {len(features)}")
    
    # Verify feature shapes
    expected_channels = encoder.get_feature_channels()
    expected_spatial_scales = [1, 0.5, 0.25, 0.125, 0.0625]  # Stage 0 no downsample, then 1-4 downsample
    
    for i, (feat, exp_channels) in enumerate(zip(features, expected_channels)):
        expected_depth = int(depth * expected_spatial_scales[i])
        expected_height = int(height * expected_spatial_scales[i])
        expected_width = int(width * expected_spatial_scales[i])
        
        expected_shape = (batch_size, exp_channels, expected_depth, expected_height, expected_width)
        
        print(f"  Stage {i}: {feat.shape} (expected: {expected_shape})")
        assert feat.shape == expected_shape, f"Stage {i} shape mismatch"
    
    # Test with different input sizes
    print("\nTesting with different input sizes...")
    
    small_x = torch.randn(1, in_channels, 32, 64, 64)
    small_features = encoder(small_x)
    print(f"Small input {small_x.shape} -> {len(small_features)} features")
    
    # Verify gradients flow
    print("\nTesting gradient flow...")
    x.requires_grad_(True)
    features = encoder(x)
    loss = sum(f.sum() for f in features)
    loss.backward()
    
    assert x.grad is not None, "Gradients not flowing to input"
    print("✓ Gradients flow correctly")
    
    print("✓ All encoder tests passed!")
    
    return encoder


if __name__ == "__main__":
    test_encoder_3d()
