"""
Query-Based Decoder for IRIS Framework

This module implements a query-based decoder that:
- Performs symmetric upsampling with skip connections
- Integrates task embeddings at each scale via cross-attention
- Produces segmentation masks
- Handles multi-class via concatenated embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskGuidedBlock3D(nn.Module):
    """
    Decoder block that integrates task embeddings via cross-attention.
    
    Args:
        in_channels (int): Number of input channels from encoder
        skip_channels (int): Number of channels from skip connection
        out_channels (int): Number of output channels
        embed_dim (int): Dimension of task embeddings
        num_heads (int): Number of attention heads (default: 8)
    """
    
    def __init__(self, in_channels, skip_channels, out_channels, embed_dim, num_heads=8):
        super().__init__()
        
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        # Channel adjustment for upsampled features
        self.up_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Channel adjustment for skip connection
        if skip_channels != out_channels:
            self.skip_conv = nn.Conv3d(skip_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip_conv = nn.Identity()
        
        # Feature processing before attention
        combined_channels = out_channels * 2  # upsampled + skip
        self.pre_attention = nn.Sequential(
            nn.Conv3d(combined_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Project spatial features to embedding dimension for attention
        self.feature_proj = nn.Conv3d(out_channels, embed_dim, kernel_size=1)
        
        # Cross-attention between spatial features and task embeddings
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Project back to feature space
        self.feature_unproj = nn.Conv3d(embed_dim, out_channels, kernel_size=1)
        
        # Final processing
        self.post_attention = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Layer normalization for attention
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, skip, task_embedding):
        """
        Forward pass of task-guided decoder block.
        
        Args:
            x: Input features from previous decoder stage (B, C_in, D, H, W)
            skip: Skip connection from encoder (B, C_skip, D*2, H*2, W*2)
            task_embedding: Task embeddings (B, num_tokens, embed_dim)
        
        Returns:
            Output features (B, C_out, D*2, H*2, W*2)
        """
        # Upsample input features
        x_up = self.upsample(x)
        x_up = self.up_conv(x_up)
        
        # Process skip connection
        skip_processed = self.skip_conv(skip)
        
        # Combine upsampled and skip features
        combined = torch.cat([x_up, skip_processed], dim=1)
        features = self.pre_attention(combined)
        
        # Apply cross-attention with task embeddings
        if task_embedding is not None:
            features = self._apply_task_attention(features, task_embedding)
        
        # Final processing
        output = self.post_attention(features)
        
        return output
    
    def _apply_task_attention(self, features, task_embedding):
        """
        Apply cross-attention between spatial features and task embeddings.
        
        Args:
            features: Spatial features (B, C, D, H, W)
            task_embedding: Task embeddings (B, num_tokens, embed_dim)
        
        Returns:
            Attended features (B, C, D, H, W)
        """
        B, C, D, H, W = features.shape
        
        # Project features to embedding dimension
        feat_proj = self.feature_proj(features)  # (B, embed_dim, D, H, W)
        
        # Reshape for attention: (B, D*H*W, embed_dim)
        feat_flat = feat_proj.view(B, self.embed_dim, -1).transpose(1, 2)
        
        # Cross-attention: spatial features attend to task embeddings
        attended_feat, _ = self.cross_attention(
            query=feat_flat,
            key=task_embedding,
            value=task_embedding
        )
        
        # Apply layer normalization and residual connection
        attended_feat = self.layer_norm(attended_feat + feat_flat)
        
        # Reshape back to spatial dimensions
        attended_feat = attended_feat.transpose(1, 2).view(B, self.embed_dim, D, H, W)
        
        # Project back to feature space
        attended_feat = self.feature_unproj(attended_feat)
        
        return attended_feat


class QueryBasedDecoder(nn.Module):
    """
    Query-based decoder with task embedding integration.
    
    Args:
        encoder_channels (list): List of encoder channel dimensions
        embed_dim (int): Dimension of task embeddings (default: 512)
        num_classes (int): Number of output classes (default: 1)
        num_heads (int): Number of attention heads (default: 8)
    """
    
    def __init__(self, encoder_channels, embed_dim=512, num_classes=1, num_heads=8):
        super().__init__()
        
        self.encoder_channels = encoder_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Decoder channels (symmetric to encoder)
        # encoder_channels: [32, 32, 64, 128, 256, 512]
        # decoder_channels: [256, 128, 64, 32, 32]
        self.decoder_channels = encoder_channels[-2::-1]  # Reverse and skip the last
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        # First block: from bottleneck (512) to first decoder stage (256)
        self.decoder_blocks.append(
            TaskGuidedBlock3D(
                in_channels=encoder_channels[-1],      # 512
                skip_channels=encoder_channels[-2],    # 256
                out_channels=self.decoder_channels[0], # 256
                embed_dim=embed_dim,
                num_heads=num_heads
            )
        )
        
        # Remaining blocks
        for i in range(1, len(self.decoder_channels)):
            skip_idx = len(encoder_channels) - 3 - i
            if skip_idx < 0:
                skip_idx = 0  # Use first encoder feature if we run out
            
            self.decoder_blocks.append(
                TaskGuidedBlock3D(
                    in_channels=self.decoder_channels[i-1],
                    skip_channels=encoder_channels[skip_idx],
                    out_channels=self.decoder_channels[i],
                    embed_dim=embed_dim,
                    num_heads=num_heads
                )
            )
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv3d(self.decoder_channels[-1], self.decoder_channels[-1], 
                     kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(self.decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.decoder_channels[-1], num_classes, kernel_size=1)
        )
        
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
    
    def forward(self, encoder_features, task_embedding=None):
        """
        Forward pass of the decoder.
        
        Args:
            encoder_features: List of encoder features [stage0, stage1, ..., stage4]
            task_embedding: Task embeddings (B, num_tokens, embed_dim) or None
        
        Returns:
            Segmentation logits (B, num_classes, D, H, W)
        """
        # Start from the bottleneck (deepest encoder features)
        x = encoder_features[-1]
        
        # Process through decoder blocks
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Get corresponding skip connection (in reverse order)
            skip_idx = len(encoder_features) - 2 - i
            if skip_idx < 0:
                skip_idx = 0  # Use first encoder feature if we run out
            skip = encoder_features[skip_idx]
            
            # Apply decoder block with task guidance
            x = decoder_block(x, skip, task_embedding)
        
        # Final output
        output = self.final_conv(x)
        
        return output


def test_decoder_3d():
    """Test function to verify Query-Based Decoder works correctly."""
    print("Testing Query-Based Decoder...")
    
    # Test parameters
    batch_size = 2
    embed_dim = 512
    num_classes = 1
    num_tokens = 10
    
    # Encoder channel configuration (matching encoder)
    encoder_channels = [32, 32, 64, 128, 256, 512]
    
    # Create mock encoder features
    base_depth, base_height, base_width = 64, 128, 128
    encoder_features = []
    
    spatial_scales = [1, 1, 0.5, 0.25, 0.125, 0.0625]
    for i, (channels, scale) in enumerate(zip(encoder_channels, spatial_scales)):
        d = int(base_depth * scale)
        h = int(base_height * scale)
        w = int(base_width * scale)
        feat = torch.randn(batch_size, channels, d, h, w)
        encoder_features.append(feat)
        print(f"  Encoder feature {i}: {feat.shape}")
    
    # Create mock task embedding
    task_embedding = torch.randn(batch_size, num_tokens, embed_dim)
    print(f"Task embedding: {task_embedding.shape}")
    
    # Create decoder
    decoder = QueryBasedDecoder(
        encoder_channels=encoder_channels,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_heads=8
    )
    
    param_count = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {param_count:,}")
    
    # Forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = decoder(encoder_features, task_embedding)
    
    expected_shape = (batch_size, num_classes, base_depth, base_height, base_width)
    print(f"Output shape: {output.shape} (expected: {expected_shape})")
    assert output.shape == expected_shape, f"Output shape mismatch"
    
    # Test without task embedding
    print("\nTesting without task embedding...")
    with torch.no_grad():
        output_no_task = decoder(encoder_features, task_embedding=None)
    
    assert output_no_task.shape == expected_shape, "Output shape mismatch without task embedding"
    print("✓ Works without task embedding")
    
    # Test multi-class
    print("\nTesting multi-class output...")
    multi_decoder = QueryBasedDecoder(
        encoder_channels=encoder_channels,
        embed_dim=embed_dim,
        num_classes=5,
        num_heads=8
    )
    
    with torch.no_grad():
        multi_output = decoder(encoder_features, task_embedding)
    
    expected_multi_shape = (batch_size, num_classes, base_depth, base_height, base_width)
    print(f"Multi-class output: {multi_output.shape}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    encoder_features[0].requires_grad_(True)
    task_embedding.requires_grad_(True)
    
    output = decoder(encoder_features, task_embedding)
    loss = output.sum()
    loss.backward()
    
    assert encoder_features[0].grad is not None, "Gradients not flowing to encoder features"
    assert task_embedding.grad is not None, "Gradients not flowing to task embedding"
    print("✓ Gradients flow correctly")
    
    print("✓ All decoder tests passed!")
    
    return decoder


if __name__ == "__main__":
    test_decoder_3d()
