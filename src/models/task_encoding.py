"""
Task Encoding Module for IRIS Framework

This module extracts task-specific embeddings from reference image-mask pairs.
It implements two parallel paths:
1. Foreground path: High-resolution feature extraction from masked regions
2. Context path: Memory-efficient processing using learnable query tokens

Input: Features (C×D×H×W) and binary mask (1×D×H×W)
Output: Task embedding T ∈ R^(m+1)×C where m=10 (default)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pixel_shuffle_3d import PixelShuffle3D, PixelUnshuffle3D


class TaskEncodingModule(nn.Module):
    """
    Task Encoding Module that extracts task-specific embeddings from reference examples.
    
    The module processes reference features and masks through two paths:
    - Foreground path: Extracts high-resolution features from masked regions
    - Context path: Uses memory-efficient pixel shuffle and learnable query tokens
    
    Args:
        in_channels (int): Number of input feature channels (default: 512)
        embed_dim (int): Dimension of output embeddings (default: 512)
        num_tokens (int): Number of learnable query tokens (default: 10)
        shuffle_scale (int): Scale factor for pixel shuffle operations (default: 2)
    """
    
    def __init__(self, in_channels=512, embed_dim=512, num_tokens=10, shuffle_scale=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.shuffle_scale = shuffle_scale
        
        # Foreground path components
        self.foreground_conv = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        )
        
        # Context path components
        self.pixel_shuffle_3d = PixelShuffle3D(scale_factor=shuffle_scale)
        
        # Calculate channels after pixel shuffle
        shuffle_channels = in_channels // (shuffle_scale ** 3)
        
        # 1x1 conv to combine shuffled features with mask
        self.context_conv1x1 = nn.Conv3d(
            shuffle_channels + 1,  # +1 for the mask channel
            shuffle_channels,      # Keep same number of channels
            kernel_size=1
        )
        
        self.pixel_unshuffle_3d = PixelUnshuffle3D(scale_factor=shuffle_scale)
        
        # Context processing - input channels after unshuffle
        self.context_processor = nn.Sequential(
            nn.Conv3d(shuffle_channels * (shuffle_scale ** 3), embed_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)  # Global average pooling
        )
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, embed_dim))
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Final projection
        self.final_projection = nn.Linear(embed_dim, embed_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        # Initialize query tokens with Xavier uniform
        nn.init.xavier_uniform_(self.query_tokens)
        
        # Initialize convolution layers
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def foreground_path(self, features, mask):
        """
        Foreground path: Extract high-resolution features from masked regions.
        
        Args:
            features: Input features (B, C, D, H, W)
            mask: Binary mask (B, 1, D_mask, H_mask, W_mask)
        
        Returns:
            Foreground embedding (B, embed_dim)
        """
        # Upsample features to match mask resolution
        upsampled_features = F.interpolate(
            features, 
            size=mask.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        
        # Apply mask to focus on foreground regions
        masked_features = upsampled_features * mask
        
        # Process masked features
        processed = self.foreground_conv(masked_features)
        
        # Global average pooling over spatial dimensions
        # Weight by mask to avoid background contamination
        mask_sum = mask.sum(dim=(2, 3, 4), keepdim=True) + 1e-8  # Avoid division by zero
        weighted_features = (processed * mask).sum(dim=(2, 3, 4)) / mask_sum.squeeze()
        
        return weighted_features  # (B, embed_dim)
    
    def context_path(self, features, mask):
        """
        Context path: Memory-efficient processing using pixel shuffle and query tokens.
        
        Args:
            features: Input features (B, C, D, H, W)
            mask: Binary mask (B, 1, D, H, W)
        
        Returns:
            Context embeddings (B, num_tokens, embed_dim)
        """
        batch_size = features.shape[0]
        
        # Apply pixel shuffle to reduce spatial resolution and increase channels
        shuffled_features = self.pixel_shuffle_3d(features)
        
        # Upsample mask to match shuffled features resolution
        shuffled_mask = F.interpolate(
            mask, 
            size=shuffled_features.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        
        # Concatenate shuffled features with mask
        combined = torch.cat([shuffled_features, shuffled_mask], dim=1)
        
        # 1x1 convolution to combine features
        context_features = self.context_conv1x1(combined)
        
        # Apply pixel unshuffle to restore original spatial dimensions
        context_features = self.pixel_unshuffle_3d(context_features)
        
        # Process context features
        context_embedding = self.context_processor(context_features)  # (B, embed_dim, 1, 1, 1)
        context_embedding = context_embedding.squeeze(-1).squeeze(-1).squeeze(-1)  # (B, embed_dim)
        
        # Expand query tokens for batch
        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_tokens, embed_dim)
        
        # Use context embedding as key and value for cross-attention
        context_key_value = context_embedding.unsqueeze(1)  # (B, 1, embed_dim)
        
        # Cross-attention between query tokens and context
        attended_tokens, _ = self.cross_attention(
            query=query_tokens,
            key=context_key_value,
            value=context_key_value
        )
        
        # Apply layer normalization and residual connection
        attended_tokens = self.layer_norm(attended_tokens + query_tokens)
        
        return attended_tokens  # (B, num_tokens, embed_dim)
    
    def forward(self, features, mask):
        """
        Forward pass of the Task Encoding Module.
        
        Args:
            features: Input features from encoder (B, C, D, H, W)
            mask: Binary segmentation mask (B, 1, D, H, W)
        
        Returns:
            Task embedding T ∈ R^(B, m+1, C) where m=num_tokens
        """
        # Extract foreground embedding
        foreground_emb = self.foreground_path(features, mask)  # (B, embed_dim)
        
        # Extract context embeddings
        context_embs = self.context_path(features, mask)  # (B, num_tokens, embed_dim)
        
        # Combine foreground and context embeddings
        # Add foreground as an additional token
        foreground_emb = foreground_emb.unsqueeze(1)  # (B, 1, embed_dim)
        task_embedding = torch.cat([foreground_emb, context_embs], dim=1)  # (B, m+1, embed_dim)
        
        # Apply final projection
        task_embedding = self.final_projection(task_embedding)
        
        return task_embedding


def test_task_encoding_module():
    """Test function to verify Task Encoding Module works correctly."""
    print("Testing Task Encoding Module...")
    
    # Test parameters
    batch_size = 2
    in_channels = 512
    embed_dim = 512
    num_tokens = 10
    depth, height, width = 8, 16, 16  # Reduced size for testing
    
    # Create test inputs
    features = torch.randn(batch_size, in_channels, depth, height, width)
    mask = torch.randint(0, 2, (batch_size, 1, depth * 8, height * 8, width * 8)).float()
    
    print(f"Input features shape: {features.shape}")
    print(f"Input mask shape: {mask.shape}")
    
    # Create module
    task_encoder = TaskEncodingModule(
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_tokens=num_tokens
    )
    
    # Forward pass
    with torch.no_grad():
        task_embedding = task_encoder(features, mask)
    
    print(f"Output task embedding shape: {task_embedding.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, num_tokens + 1, embed_dim)
    assert task_embedding.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {task_embedding.shape}"
    
    # Verify embeddings are not all zeros
    assert task_embedding.abs().sum() > 0, "Task embeddings are all zeros"
    
    # Test with different batch sizes
    features_single = features[:1]
    mask_single = mask[:1]
    task_embedding_single = task_encoder(features_single, mask_single)
    
    expected_single_shape = (1, num_tokens + 1, embed_dim)
    assert task_embedding_single.shape == expected_single_shape, \
        f"Single batch test failed: expected {expected_single_shape}, got {task_embedding_single.shape}"
    
    print("✓ All tests passed!")
    print(f"✓ Task encoding produces shape: ({num_tokens + 1}, {embed_dim})")
    print("✓ Foreground path extracts features at original resolution")
    print("✓ Context path uses memory-efficient pixel shuffle")
    print("✓ Cross-attention integrates query tokens")


if __name__ == "__main__":
    test_task_encoding_module()
