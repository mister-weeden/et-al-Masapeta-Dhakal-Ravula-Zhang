"""
3D PixelShuffle Implementation for IRIS Framework

The paper mentions "strategy similar to sub-pixel convolution" but PyTorch only has 2D.
This implements custom 3D pixel shuffle operations for memory-efficient processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffle3D(nn.Module):
    """
    3D version of pixel shuffle operation.
    
    Rearranges channels to spatial dimensions:
    Input: (B, C×r³, D, H, W)
    Output: (B, C, D×r, H×r, W×r)
    
    Args:
        scale_factor (int): Upsampling factor for each spatial dimension
    """
    
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.scale_factor_cubed = scale_factor ** 3
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C×r³, D, H, W)
        
        Returns:
            Output tensor of shape (B, C, D×r, H×r, W×r)
        """
        batch_size, channels, depth, height, width = x.shape
        
        # Ensure channels is divisible by scale_factor³
        assert channels % self.scale_factor_cubed == 0, \
            f"Channels ({channels}) must be divisible by scale_factor³ ({self.scale_factor_cubed})"
        
        out_channels = channels // self.scale_factor_cubed
        
        # Reshape to separate the scale factors
        # (B, C×r³, D, H, W) -> (B, C, r, r, r, D, H, W)
        x = x.view(batch_size, out_channels, 
                   self.scale_factor, self.scale_factor, self.scale_factor,
                   depth, height, width)
        
        # Permute to interleave spatial dimensions
        # (B, C, r, r, r, D, H, W) -> (B, C, D, r, H, r, W, r)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        
        # Reshape to final output
        # (B, C, D, r, H, r, W, r) -> (B, C, D×r, H×r, W×r)
        x = x.view(batch_size, out_channels,
                   depth * self.scale_factor,
                   height * self.scale_factor,
                   width * self.scale_factor)
        
        return x


class PixelUnshuffle3D(nn.Module):
    """
    3D version of pixel unshuffle operation (inverse of PixelShuffle3D).
    
    Rearranges spatial dimensions to channels:
    Input: (B, C, D×r, H×r, W×r)
    Output: (B, C×r³, D, H, W)
    
    Args:
        scale_factor (int): Downsampling factor for each spatial dimension
    """
    
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.scale_factor_cubed = scale_factor ** 3
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, D×r, H×r, W×r)
        
        Returns:
            Output tensor of shape (B, C×r³, D, H, W)
        """
        batch_size, channels, depth, height, width = x.shape
        
        # Ensure spatial dimensions are divisible by scale_factor
        assert depth % self.scale_factor == 0, \
            f"Depth ({depth}) must be divisible by scale_factor ({self.scale_factor})"
        assert height % self.scale_factor == 0, \
            f"Height ({height}) must be divisible by scale_factor ({self.scale_factor})"
        assert width % self.scale_factor == 0, \
            f"Width ({width}) must be divisible by scale_factor ({self.scale_factor})"
        
        out_depth = depth // self.scale_factor
        out_height = height // self.scale_factor
        out_width = width // self.scale_factor
        
        # Reshape to separate the scale factors
        # (B, C, D×r, H×r, W×r) -> (B, C, D, r, H, r, W, r)
        x = x.view(batch_size, channels,
                   out_depth, self.scale_factor,
                   out_height, self.scale_factor,
                   out_width, self.scale_factor)
        
        # Permute to group scale factors
        # (B, C, D, r, H, r, W, r) -> (B, C, r, r, r, D, H, W)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        
        # Reshape to final output
        # (B, C, r, r, r, D, H, W) -> (B, C×r³, D, H, W)
        x = x.view(batch_size, channels * self.scale_factor_cubed,
                   out_depth, out_height, out_width)
        
        return x


def test_pixel_shuffle_3d():
    """Test function to verify 3D pixel shuffle operations work correctly."""
    print("Testing 3D PixelShuffle operations...")
    
    # Test parameters
    batch_size = 2
    channels = 64
    depth, height, width = 16, 32, 32
    scale_factor = 2
    
    # Create test input
    x = torch.randn(batch_size, channels * (scale_factor ** 3), depth, height, width)
    print(f"Input shape: {x.shape}")
    
    # Test PixelShuffle3D
    shuffle = PixelShuffle3D(scale_factor)
    shuffled = shuffle(x)
    print(f"After PixelShuffle3D: {shuffled.shape}")
    
    expected_shape = (batch_size, channels, 
                     depth * scale_factor, 
                     height * scale_factor, 
                     width * scale_factor)
    assert shuffled.shape == expected_shape, f"Expected {expected_shape}, got {shuffled.shape}"
    
    # Test PixelUnshuffle3D (inverse operation)
    unshuffle = PixelUnshuffle3D(scale_factor)
    unshuffled = unshuffle(shuffled)
    print(f"After PixelUnshuffle3D: {unshuffled.shape}")
    
    assert unshuffled.shape == x.shape, f"Expected {x.shape}, got {unshuffled.shape}"
    
    # Test that operations are inverses (within numerical precision)
    diff = torch.abs(x - unshuffled).max()
    print(f"Max difference after round-trip: {diff.item()}")
    assert diff < 1e-6, f"Operations are not inverses, max diff: {diff.item()}"
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_pixel_shuffle_3d()
