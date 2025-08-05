"""
Complete IRIS Model for Universal Medical Image Segmentation

This module integrates all components:
- 3D UNet Encoder: Extracts multi-scale features
- Task Encoding Module: Generates task-specific embeddings from reference examples
- Query-Based Decoder: Produces segmentation guided by task embeddings

The model performs in-context learning for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_3d import Encoder3D
from .task_encoding import TaskEncodingModule
from .decoder_3d import QueryBasedDecoder


class IRISModel(nn.Module):
    """
    Complete IRIS model for universal medical image segmentation.
    
    The model takes a query image and reference image-mask pairs to perform
    segmentation without fine-tuning, using in-context learning.
    
    Args:
        in_channels (int): Number of input image channels (default: 1)
        base_channels (int): Base number of encoder channels (default: 32)
        embed_dim (int): Task embedding dimension (default: 512)
        num_tokens (int): Number of query tokens in task encoding (default: 10)
        num_classes (int): Number of output classes (default: 1)
        num_heads (int): Number of attention heads (default: 8)
        shuffle_scale (int): Scale factor for pixel shuffle in task encoding (default: 2)
    """
    
    def __init__(self, in_channels=1, base_channels=32, embed_dim=512, 
                 num_tokens=10, num_classes=1, num_heads=8, shuffle_scale=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        
        # Image encoder (shared for both reference and query images)
        self.encoder = Encoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks_per_stage=2
        )
        
        # Get encoder channel configuration
        encoder_channels = self.encoder.get_feature_channels()
        
        # Task encoding module
        self.task_encoder = TaskEncodingModule(
            in_channels=encoder_channels[-1],  # Use bottleneck features (512)
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            shuffle_scale=shuffle_scale
        )
        
        # Query-based decoder
        self.decoder = QueryBasedDecoder(
            encoder_channels=encoder_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_heads=num_heads
        )
        
        # Store configuration
        self.config = {
            'in_channels': in_channels,
            'base_channels': base_channels,
            'embed_dim': embed_dim,
            'num_tokens': num_tokens,
            'num_classes': num_classes,
            'num_heads': num_heads,
            'shuffle_scale': shuffle_scale,
            'encoder_channels': encoder_channels
        }
    
    def encode_image(self, image):
        """
        Encode an image through the encoder.
        
        Args:
            image: Input image (B, C, D, H, W)
        
        Returns:
            List of multi-scale features
        """
        return self.encoder(image)
    
    def encode_task(self, reference_image, reference_mask):
        """
        Encode task from reference image-mask pair.
        
        Args:
            reference_image: Reference image (B, C, D, H, W)
            reference_mask: Reference mask (B, 1, D, H, W)
        
        Returns:
            Task embedding (B, num_tokens+1, embed_dim)
        """
        # Extract features from reference image
        ref_features = self.encode_image(reference_image)
        
        # Use bottleneck features for task encoding
        bottleneck_features = ref_features[-1]
        
        # Generate task embedding
        task_embedding = self.task_encoder(bottleneck_features, reference_mask)
        
        return task_embedding
    
    def segment_with_task(self, query_image, task_embedding):
        """
        Segment query image using task embedding.
        
        Args:
            query_image: Query image to segment (B, C, D, H, W)
            task_embedding: Task embedding (B, num_tokens+1, embed_dim)
        
        Returns:
            Segmentation logits (B, num_classes, D, H, W)
        """
        # Extract features from query image
        query_features = self.encode_image(query_image)
        
        # Decode with task guidance
        segmentation = self.decoder(query_features, task_embedding)
        
        return segmentation
    
    def forward(self, query_image, reference_image=None, reference_mask=None, task_embedding=None):
        """
        Forward pass of IRIS model.
        
        Two modes of operation:
        1. End-to-end: Provide query_image, reference_image, and reference_mask
        2. With pre-computed task: Provide query_image and task_embedding
        
        Args:
            query_image: Query image to segment (B, C, D, H, W)
            reference_image: Reference image (B, C, D, H, W) [optional]
            reference_mask: Reference mask (B, 1, D, H, W) [optional]
            task_embedding: Pre-computed task embedding (B, num_tokens+1, embed_dim) [optional]
        
        Returns:
            Segmentation logits (B, num_classes, D, H, W)
        """
        if task_embedding is None:
            if reference_image is None or reference_mask is None:
                raise ValueError("Must provide either task_embedding or both reference_image and reference_mask")
            
            # Encode task from reference
            task_embedding = self.encode_task(reference_image, reference_mask)
        
        # Segment query image
        segmentation = self.segment_with_task(query_image, task_embedding)
        
        return segmentation
    
    def get_model_info(self):
        """Get model information and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        task_encoder_params = sum(p.numel() for p in self.task_encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_parameters': encoder_params,
            'task_encoder_parameters': task_encoder_params,
            'decoder_parameters': decoder_params,
            'config': self.config
        }
        
        return info


class IRISInference:
    """
    Inference utilities for IRIS model.
    
    Provides methods for:
    - One-shot inference with single reference
    - Memory bank for storing task embeddings
    - Sliding window inference for large volumes
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.task_memory = {}  # Store task embeddings by class name
        
        self.model.to(device)
        self.model.eval()
    
    def one_shot_inference(self, query_image, reference_image, reference_mask, 
                          apply_sigmoid=True, threshold=0.5):
        """
        Perform one-shot inference with a single reference example.
        
        Args:
            query_image: Query image tensor (1, C, D, H, W)
            reference_image: Reference image tensor (1, C, D, H, W)
            reference_mask: Reference mask tensor (1, 1, D, H, W)
            apply_sigmoid: Whether to apply sigmoid to output (default: True)
            threshold: Threshold for binary prediction (default: 0.5)
        
        Returns:
            Dictionary with 'logits', 'probabilities', and 'prediction'
        """
        with torch.no_grad():
            # Move to device
            query_image = query_image.to(self.device)
            reference_image = reference_image.to(self.device)
            reference_mask = reference_mask.to(self.device)
            
            # Forward pass
            logits = self.model(query_image, reference_image, reference_mask)
            
            # Post-process
            if apply_sigmoid:
                probabilities = torch.sigmoid(logits)
                prediction = (probabilities > threshold).float()
            else:
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1, keepdim=True).float()
            
            return {
                'logits': logits.cpu(),
                'probabilities': probabilities.cpu(),
                'prediction': prediction.cpu()
            }
    
    def store_task_embedding(self, class_name, reference_image, reference_mask):
        """
        Store task embedding for a class in memory bank.
        
        Args:
            class_name: Name of the anatomical class
            reference_image: Reference image tensor (1, C, D, H, W)
            reference_mask: Reference mask tensor (1, 1, D, H, W)
        """
        with torch.no_grad():
            reference_image = reference_image.to(self.device)
            reference_mask = reference_mask.to(self.device)
            
            task_embedding = self.model.encode_task(reference_image, reference_mask)
            self.task_memory[class_name] = task_embedding.cpu()
    
    def inference_with_stored_task(self, query_image, class_name, 
                                  apply_sigmoid=True, threshold=0.5):
        """
        Perform inference using stored task embedding.
        
        Args:
            query_image: Query image tensor (1, C, D, H, W)
            class_name: Name of the stored class
            apply_sigmoid: Whether to apply sigmoid to output (default: True)
            threshold: Threshold for binary prediction (default: 0.5)
        
        Returns:
            Dictionary with 'logits', 'probabilities', and 'prediction'
        """
        if class_name not in self.task_memory:
            raise ValueError(f"Class '{class_name}' not found in task memory")
        
        with torch.no_grad():
            query_image = query_image.to(self.device)
            task_embedding = self.task_memory[class_name].to(self.device)
            
            logits = self.model.segment_with_task(query_image, task_embedding)
            
            # Post-process
            if apply_sigmoid:
                probabilities = torch.sigmoid(logits)
                prediction = (probabilities > threshold).float()
            else:
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1, keepdim=True).float()
            
            return {
                'logits': logits.cpu(),
                'probabilities': probabilities.cpu(),
                'prediction': prediction.cpu()
            }


def test_iris_model():
    """Test function to verify complete IRIS model works correctly."""
    print("Testing Complete IRIS Model...")
    
    # Test parameters
    batch_size = 1  # Reduced for memory
    in_channels = 1
    base_channels = 16  # Reduced for testing
    embed_dim = 64     # Reduced for testing
    num_tokens = 5     # Reduced for testing
    num_classes = 1
    
    # Spatial dimensions (small for testing)
    depth, height, width = 32, 64, 64
    
    print(f"Test configuration:")
    print(f"  - Input shape: ({batch_size}, {in_channels}, {depth}, {height}, {width})")
    print(f"  - Base channels: {base_channels}")
    print(f"  - Embed dim: {embed_dim}")
    print(f"  - Num tokens: {num_tokens}")
    
    # Create model
    model = IRISModel(
        in_channels=in_channels,
        base_channels=base_channels,
        embed_dim=embed_dim,
        num_tokens=num_tokens,
        num_classes=num_classes
    )
    
    # Print model info
    info = model.get_model_info()
    print(f"\nModel parameters:")
    print(f"  - Total: {info['total_parameters']:,}")
    print(f"  - Encoder: {info['encoder_parameters']:,}")
    print(f"  - Task Encoder: {info['task_encoder_parameters']:,}")
    print(f"  - Decoder: {info['decoder_parameters']:,}")
    
    # Create test inputs
    query_image = torch.randn(batch_size, in_channels, depth, height, width)
    reference_image = torch.randn(batch_size, in_channels, depth, height, width)
    reference_mask = torch.randint(0, 2, (batch_size, 1, depth, height, width)).float()
    
    print(f"\nTest inputs:")
    print(f"  - Query image: {query_image.shape}")
    print(f"  - Reference image: {reference_image.shape}")
    print(f"  - Reference mask: {reference_mask.shape}, coverage: {reference_mask.mean():.3f}")
    
    # Test end-to-end forward pass
    print("\n1. Testing end-to-end forward pass...")
    with torch.no_grad():
        output = model(query_image, reference_image, reference_mask)
    
    expected_shape = (batch_size, num_classes, depth, height, width)
    print(f"   Output shape: {output.shape} (expected: {expected_shape})")
    assert output.shape == expected_shape, "Output shape mismatch"
    print("   ✓ End-to-end forward pass successful")
    
    # Test two-stage inference
    print("\n2. Testing two-stage inference...")
    with torch.no_grad():
        # Stage 1: Encode task
        task_embedding = model.encode_task(reference_image, reference_mask)
        print(f"   Task embedding: {task_embedding.shape}")
        
        # Stage 2: Segment with task
        output2 = model.segment_with_task(query_image, task_embedding)
        print(f"   Segmentation output: {output2.shape}")
        
        # Should be identical to end-to-end
        diff = torch.abs(output - output2).max()
        print(f"   Difference from end-to-end: {diff.item()}")
        assert diff < 1e-6, "Two-stage inference differs from end-to-end"
    
    print("   ✓ Two-stage inference matches end-to-end")
    
    # Test inference utilities
    print("\n3. Testing inference utilities...")
    inference = IRISInference(model, device='cpu')
    
    # One-shot inference
    result = inference.one_shot_inference(
        query_image, reference_image, reference_mask
    )
    
    print(f"   One-shot result keys: {list(result.keys())}")
    print(f"   Prediction shape: {result['prediction'].shape}")
    print(f"   Prediction range: [{result['probabilities'].min():.3f}, {result['probabilities'].max():.3f}]")
    
    # Store and retrieve task
    inference.store_task_embedding('test_organ', reference_image, reference_mask)
    stored_result = inference.inference_with_stored_task(query_image, 'test_organ')
    
    # Should be identical
    pred_diff = torch.abs(result['prediction'] - stored_result['prediction']).max()
    print(f"   Stored task difference: {pred_diff.item()}")
    assert pred_diff < 1e-6, "Stored task inference differs"
    
    print("   ✓ Inference utilities work correctly")
    
    # Test gradient flow
    print("\n4. Testing gradient flow...")
    query_image.requires_grad_(True)
    reference_image.requires_grad_(True)
    reference_mask.requires_grad_(True)
    
    output = model(query_image, reference_image, reference_mask)
    loss = output.sum()
    loss.backward()
    
    assert query_image.grad is not None, "No gradients to query image"
    assert reference_image.grad is not None, "No gradients to reference image"
    print("   ✓ Gradients flow correctly")
    
    print("\n🎉 All IRIS model tests passed!")
    print("\nPhase 2 Implementation Complete!")
    
    return model, inference


if __name__ == "__main__":
    test_iris_model()
