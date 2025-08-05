"""
Fixed Query-Based Decoder for IRIS Framework

This module fixes the channel mismatch issues in the original decoder
by properly handling skip connection dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskGuidedBlock3D(nn.Module):
    """
    Fixed decoder block that integrates task embeddings via cross-attention.
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
        self.skip_conv = nn.Conv3d(skip_channels, out_channels, kernel_size=1, bias=False)
        
        # Feature processing after combining
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
        """Apply cross-attention between spatial features and task embeddings."""
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


class FixedQueryBasedDecoder(nn.Module):
    """
    Fixed query-based decoder with proper channel handling.
    """
    
    def __init__(self, encoder_channels, embed_dim=512, num_classes=1, num_heads=8):
        super().__init__()
        
        self.encoder_channels = encoder_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Define decoder channel progression
        # encoder_channels: [32, 32, 64, 128, 256, 512] (example)
        # We'll decode from 512 -> 256 -> 128 -> 64 -> 32
        
        self.decoder_channels = [256, 128, 64, 32]
        
        # Create decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        # Block 0: 512 -> 256 (with skip from 256)
        self.decoder_blocks.append(
            TaskGuidedBlock3D(
                in_channels=encoder_channels[-1],    # 512
                skip_channels=encoder_channels[-2],  # 256
                out_channels=self.decoder_channels[0], # 256
                embed_dim=embed_dim,
                num_heads=num_heads
            )
        )
        
        # Block 1: 256 -> 128 (with skip from 128)
        self.decoder_blocks.append(
            TaskGuidedBlock3D(
                in_channels=self.decoder_channels[0], # 256
                skip_channels=encoder_channels[-3],   # 128
                out_channels=self.decoder_channels[1], # 128
                embed_dim=embed_dim,
                num_heads=num_heads
            )
        )
        
        # Block 2: 128 -> 64 (with skip from 64)
        self.decoder_blocks.append(
            TaskGuidedBlock3D(
                in_channels=self.decoder_channels[1], # 128
                skip_channels=encoder_channels[-4],   # 64
                out_channels=self.decoder_channels[2], # 64
                embed_dim=embed_dim,
                num_heads=num_heads
            )
        )
        
        # Block 3: 64 -> 32 (with skip from 32)
        self.decoder_blocks.append(
            TaskGuidedBlock3D(
                in_channels=self.decoder_channels[2], # 64
                skip_channels=encoder_channels[-5],   # 32
                out_channels=self.decoder_channels[3], # 32
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
        Forward pass of the fixed decoder.
        
        Args:
            encoder_features: List of encoder features [stage0, stage1, ..., stage4]
            task_embedding: Task embeddings (B, num_tokens, embed_dim) or None
        
        Returns:
            Segmentation logits (B, num_classes, D, H, W)
        """
        # Start from the bottleneck (deepest encoder features)
        x = encoder_features[-1]  # 512 channels
        
        # Process through decoder blocks with proper skip connections
        skip_indices = [-2, -3, -4, -5]  # Corresponding encoder features
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = encoder_features[skip_indices[i]]
            x = decoder_block(x, skip, task_embedding)
        
        # Final output
        output = self.final_conv(x)
        
        return output


def test_fixed_decoder():
    """Test the fixed decoder implementation."""
    print("Testing Fixed Query-Based Decoder...")
    
    # Test parameters
    batch_size = 1
    embed_dim = 64
    num_classes = 1
    num_tokens = 5
    
    # Encoder channel configuration (matching our encoder)
    encoder_channels = [8, 8, 16, 32, 64, 128]  # Small for testing
    
    # Create mock encoder features with correct shapes
    base_depth, base_height, base_width = 16, 32, 32
    encoder_features = []
    
    spatial_scales = [1, 0.5, 0.25, 0.125, 0.0625]
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
    
    # Create fixed decoder
    decoder = FixedQueryBasedDecoder(
        encoder_channels=encoder_channels,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_heads=4
    )
    
    param_count = sum(p.numel() for p in decoder.parameters())
    print(f"Fixed decoder parameters: {param_count:,}")
    
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
    
    print("\n✅ Fixed decoder tests passed!")
    
    return decoder


if __name__ == "__main__":
    test_fixed_decoder()
