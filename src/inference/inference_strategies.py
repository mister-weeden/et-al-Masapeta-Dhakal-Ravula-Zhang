"""
Phase 4: Inference Strategies for IRIS Framework

This module implements various inference strategies for the IRIS model:
1. One-Shot Inference: Single reference example segmentation
2. Memory Bank: Task embedding storage and retrieval
3. Sliding Window: Large volume processing
4. Multi-Class Inference: Simultaneous multi-organ segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import pickle
from collections import defaultdict
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.iris_model import IRISModel


class TaskMemoryBank:
    """
    Memory bank for storing and retrieving task embeddings.
    
    Stores task embeddings by class name with EMA updates during training.
    Enables fast inference on seen classes without recomputing embeddings.
    
    Args:
        momentum (float): EMA momentum for updating embeddings (default: 0.999)
        device (str): Device to store embeddings on (default: 'cuda')
    """
    
    def __init__(self, momentum: float = 0.999, device: str = 'cuda'):
        self.momentum = momentum
        self.device = device
        self.embeddings = {}  # class_name -> embedding tensor
        self.counts = {}      # class_name -> update count
        self.metadata = {}    # class_name -> metadata dict
    
    def store_embedding(self, class_name: str, embedding: torch.Tensor, 
                       dataset_name: str = None, confidence: float = 1.0):
        """
        Store or update task embedding for a class.
        
        Args:
            class_name: Name of the anatomical class
            embedding: Task embedding tensor (1, num_tokens, embed_dim)
            dataset_name: Source dataset name (optional)
            confidence: Confidence score for this embedding (default: 1.0)
        """
        embedding = embedding.to(self.device).detach()
        
        if class_name in self.embeddings:
            # EMA update
            old_embedding = self.embeddings[class_name]
            new_embedding = self.momentum * old_embedding + (1 - self.momentum) * embedding
            self.embeddings[class_name] = new_embedding
            self.counts[class_name] += 1
        else:
            # First time storing
            self.embeddings[class_name] = embedding.clone()
            self.counts[class_name] = 1
        
        # Update metadata
        self.metadata[class_name] = {
            'dataset_name': dataset_name,
            'confidence': confidence,
            'last_updated': time.time(),
            'update_count': self.counts[class_name]
        }
    
    def retrieve_embedding(self, class_name: str) -> Optional[torch.Tensor]:
        """
        Retrieve task embedding for a class.
        
        Args:
            class_name: Name of the anatomical class
        
        Returns:
            Task embedding tensor or None if not found
        """
        if class_name in self.embeddings:
            return self.embeddings[class_name].clone()
        return None
    
    def get_similar_classes(self, class_name: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find similar classes based on embedding similarity.
        
        Args:
            class_name: Query class name
            top_k: Number of similar classes to return
        
        Returns:
            List of (class_name, similarity_score) tuples
        """
        if class_name not in self.embeddings:
            return []
        
        query_embedding = self.embeddings[class_name]
        similarities = []
        
        for other_class, other_embedding in self.embeddings.items():
            if other_class != class_name:
                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    query_embedding.flatten(), 
                    other_embedding.flatten(), 
                    dim=0
                ).item()
                similarities.append((other_class, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save(self, filepath: str):
        """Save memory bank to file."""
        data = {
            'embeddings': {k: v.cpu() for k, v in self.embeddings.items()},
            'counts': self.counts,
            'metadata': self.metadata,
            'momentum': self.momentum
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load memory bank from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = {k: v.to(self.device) for k, v in data['embeddings'].items()}
        self.counts = data['counts']
        self.metadata = data['metadata']
        self.momentum = data.get('momentum', 0.999)
    
    def get_stats(self) -> Dict:
        """Get memory bank statistics."""
        return {
            'num_classes': len(self.embeddings),
            'total_updates': sum(self.counts.values()),
            'classes': list(self.embeddings.keys()),
            'embedding_shape': list(self.embeddings.values())[0].shape if self.embeddings else None
        }


class SlidingWindowInference:
    """
    Sliding window inference for processing large medical volumes.
    
    Processes large volumes by dividing them into overlapping patches,
    running inference on each patch, and combining results.
    
    Args:
        patch_size: Size of each patch (D, H, W)
        overlap: Overlap ratio between patches (default: 0.5)
        batch_size: Number of patches to process simultaneously (default: 1)
    """
    
    def __init__(self, patch_size: Tuple[int, int, int] = (64, 128, 128),
                 overlap: float = 0.5, batch_size: int = 1):
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.stride = tuple(int(s * (1 - overlap)) for s in patch_size)
    
    def extract_patches(self, volume: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Extract overlapping patches from volume.
        
        Args:
            volume: Input volume (C, D, H, W)
        
        Returns:
            patches: Tensor of patches (N, C, patch_D, patch_H, patch_W)
            coordinates: List of patch coordinates
        """
        C, D, H, W = volume.shape
        patch_D, patch_H, patch_W = self.patch_size
        stride_D, stride_H, stride_W = self.stride
        
        patches = []
        coordinates = []
        
        # Generate patch coordinates
        for d in range(0, D - patch_D + 1, stride_D):
            for h in range(0, H - patch_H + 1, stride_H):
                for w in range(0, W - patch_W + 1, stride_W):
                    # Ensure patch doesn't exceed volume bounds
                    d_end = min(d + patch_D, D)
                    h_end = min(h + patch_H, H)
                    w_end = min(w + patch_W, W)
                    
                    d_start = d_end - patch_D
                    h_start = h_end - patch_H
                    w_start = w_end - patch_W
                    
                    patch = volume[:, d_start:d_end, h_start:h_end, w_start:w_end]
                    patches.append(patch)
                    coordinates.append((d_start, h_start, w_start, d_end, h_end, w_end))
        
        if patches:
            patches = torch.stack(patches, dim=0)
        else:
            # If volume is smaller than patch size, use the whole volume
            patches = volume.unsqueeze(0)
            coordinates = [(0, 0, 0, D, H, W)]
        
        return patches, coordinates
    
    def combine_patches(self, patch_predictions: torch.Tensor, 
                       coordinates: List[Tuple], output_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Combine patch predictions into full volume.
        
        Args:
            patch_predictions: Predictions for each patch (N, num_classes, patch_D, patch_H, patch_W)
            coordinates: Patch coordinates
            output_shape: Shape of output volume (num_classes, D, H, W)
        
        Returns:
            Combined prediction volume
        """
        num_classes, D, H, W = output_shape
        
        # Initialize output volume and weight map
        output = torch.zeros(output_shape, device=patch_predictions.device)
        weight_map = torch.zeros((D, H, W), device=patch_predictions.device)
        
        # Create Gaussian weight for each patch (higher weight in center)
        patch_D, patch_H, patch_W = self.patch_size
        gaussian_weight = self._create_gaussian_weight(patch_D, patch_H, patch_W)
        gaussian_weight = gaussian_weight.to(patch_predictions.device)
        
        # Combine patches
        for i, (d_start, h_start, w_start, d_end, h_end, w_end) in enumerate(coordinates):
            pred = patch_predictions[i]  # (num_classes, patch_D, patch_H, patch_W)
            
            # Add weighted prediction
            output[:, d_start:d_end, h_start:h_end, w_start:w_end] += pred * gaussian_weight
            weight_map[d_start:d_end, h_start:h_end, w_start:w_end] += gaussian_weight
        
        # Normalize by weights
        weight_map = torch.clamp(weight_map, min=1e-8)
        output = output / weight_map.unsqueeze(0)
        
        return output
    
    def _create_gaussian_weight(self, D: int, H: int, W: int) -> torch.Tensor:
        """Create Gaussian weight map for patch blending."""
        # Create 1D Gaussian for each dimension
        def gaussian_1d(size, sigma=None):
            if sigma is None:
                sigma = size / 6.0  # 3-sigma rule
            coords = torch.arange(size, dtype=torch.float32)
            coords = coords - (size - 1) / 2.0
            return torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        
        weight_d = gaussian_1d(D)
        weight_h = gaussian_1d(H)
        weight_w = gaussian_1d(W)
        
        # Create 3D Gaussian weight
        weight_3d = weight_d.view(-1, 1, 1) * weight_h.view(1, -1, 1) * weight_w.view(1, 1, -1)
        
        return weight_3d


class MultiClassInference:
    """
    Multi-class inference for simultaneous multi-organ segmentation.
    
    Enables segmentation of multiple anatomical structures in a single
    forward pass using multiple task embeddings.
    """
    
    def __init__(self, model: IRISModel, memory_bank: TaskMemoryBank):
        self.model = model
        self.memory_bank = memory_bank
        self.model.eval()
    
    def segment_multiple_classes(self, query_image: torch.Tensor, 
                                class_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Segment multiple classes simultaneously.
        
        Args:
            query_image: Query image to segment (1, C, D, H, W)
            class_names: List of class names to segment
        
        Returns:
            Dictionary mapping class names to segmentation predictions
        """
        results = {}
        
        with torch.no_grad():
            # Extract query features once
            query_features = self.model.encode_image(query_image)
            
            # Process each class
            for class_name in class_names:
                task_embedding = self.memory_bank.retrieve_embedding(class_name)
                
                if task_embedding is not None:
                    # Segment with stored task embedding
                    prediction = self.model.decoder(query_features, task_embedding)
                    results[class_name] = torch.sigmoid(prediction)
                else:
                    print(f"Warning: No task embedding found for class '{class_name}'")
                    # Return zero prediction
                    results[class_name] = torch.zeros(
                        1, 1, *query_image.shape[2:], 
                        device=query_image.device
                    )
        
        return results
    
    def segment_with_confidence(self, query_image: torch.Tensor, 
                               class_names: List[str], 
                               confidence_threshold: float = 0.5) -> Dict[str, Dict]:
        """
        Segment multiple classes with confidence estimation.
        
        Args:
            query_image: Query image to segment
            class_names: List of class names to segment
            confidence_threshold: Threshold for confident predictions
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        predictions = self.segment_multiple_classes(query_image, class_names)
        results = {}
        
        for class_name, prediction in predictions.items():
            # Compute confidence metrics
            max_prob = prediction.max().item()
            mean_prob = prediction.mean().item()
            volume_ratio = (prediction > confidence_threshold).float().mean().item()
            
            results[class_name] = {
                'prediction': prediction,
                'max_confidence': max_prob,
                'mean_confidence': mean_prob,
                'volume_ratio': volume_ratio,
                'is_confident': max_prob > confidence_threshold
            }
        
        return results


class IRISInferenceEngine:
    """
    Complete inference engine for IRIS model.
    
    Combines all inference strategies into a unified interface.
    """
    
    def __init__(self, model: IRISModel, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Initialize components
        self.memory_bank = TaskMemoryBank(device=device)
        self.sliding_window = SlidingWindowInference()
        self.multi_class = MultiClassInference(model, self.memory_bank)
    
    def one_shot_inference(self, query_image: torch.Tensor, 
                          reference_image: torch.Tensor, 
                          reference_mask: torch.Tensor,
                          use_sliding_window: bool = False) -> Dict[str, torch.Tensor]:
        """
        Perform one-shot inference with single reference example.
        
        Args:
            query_image: Query image to segment (1, C, D, H, W)
            reference_image: Reference image (1, C, D, H, W)
            reference_mask: Reference mask (1, 1, D, H, W)
            use_sliding_window: Whether to use sliding window for large volumes
        
        Returns:
            Dictionary with prediction results
        """
        with torch.no_grad():
            # Move to device
            query_image = query_image.to(self.device)
            reference_image = reference_image.to(self.device)
            reference_mask = reference_mask.to(self.device)
            
            if use_sliding_window and self._is_large_volume(query_image):
                return self._sliding_window_inference(
                    query_image, reference_image, reference_mask
                )
            else:
                # Standard inference
                task_embedding = self.model.encode_task(reference_image, reference_mask)
                prediction = self.model.segment_with_task(query_image, task_embedding)
                
                return {
                    'logits': prediction,
                    'probabilities': torch.sigmoid(prediction),
                    'prediction': (torch.sigmoid(prediction) > 0.5).float(),
                    'task_embedding': task_embedding
                }
    
    def memory_bank_inference(self, query_image: torch.Tensor, 
                             class_name: str) -> Dict[str, torch.Tensor]:
        """
        Perform inference using stored task embedding.
        
        Args:
            query_image: Query image to segment
            class_name: Name of the anatomical class
        
        Returns:
            Dictionary with prediction results
        """
        task_embedding = self.memory_bank.retrieve_embedding(class_name)
        
        if task_embedding is None:
            raise ValueError(f"No task embedding found for class '{class_name}'")
        
        with torch.no_grad():
            query_image = query_image.to(self.device)
            prediction = self.model.segment_with_task(query_image, task_embedding)
            
            return {
                'logits': prediction,
                'probabilities': torch.sigmoid(prediction),
                'prediction': (torch.sigmoid(prediction) > 0.5).float(),
                'class_name': class_name
            }
    
    def _is_large_volume(self, volume: torch.Tensor, threshold: int = 256) -> bool:
        """Check if volume is large enough to benefit from sliding window."""
        return any(s > threshold for s in volume.shape[2:])
    
    def _sliding_window_inference(self, query_image: torch.Tensor,
                                 reference_image: torch.Tensor,
                                 reference_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform sliding window inference for large volumes."""
        # Extract task embedding once
        task_embedding = self.model.encode_task(reference_image, reference_mask)
        
        # Extract patches from query image
        query_patches, coordinates = self.sliding_window.extract_patches(query_image.squeeze(0))
        
        # Process patches in batches
        patch_predictions = []
        batch_size = self.sliding_window.batch_size
        
        for i in range(0, len(query_patches), batch_size):
            batch_patches = query_patches[i:i+batch_size]
            
            with torch.no_grad():
                batch_predictions = []
                for patch in batch_patches:
                    patch = patch.unsqueeze(0)  # Add batch dimension
                    pred = self.model.segment_with_task(patch, task_embedding)
                    batch_predictions.append(pred.squeeze(0))
                
                batch_predictions = torch.stack(batch_predictions)
                patch_predictions.append(batch_predictions)
        
        # Combine all patch predictions
        all_predictions = torch.cat(patch_predictions, dim=0)
        
        # Combine patches into full volume
        output_shape = (1, *query_image.shape[2:])  # (num_classes, D, H, W)
        combined_prediction = self.sliding_window.combine_patches(
            all_predictions, coordinates, output_shape
        )
        
        return {
            'logits': combined_prediction.unsqueeze(0),  # Add batch dimension
            'probabilities': torch.sigmoid(combined_prediction.unsqueeze(0)),
            'prediction': (torch.sigmoid(combined_prediction) > 0.5).float().unsqueeze(0),
            'task_embedding': task_embedding
        }


def test_inference_strategies():
    """Test all inference strategies."""
    print("Testing IRIS Inference Strategies...")
    
    # Create test model and data
    model = IRISModel(
        in_channels=1, base_channels=4, embed_dim=16, 
        num_tokens=2, num_classes=1
    )
    
    # Test data
    query_image = torch.randn(1, 1, 16, 32, 32)
    reference_image = torch.randn(1, 1, 16, 32, 32)
    reference_mask = torch.randint(0, 2, (1, 1, 16, 32, 32)).float()
    
    print(f"Test data shapes:")
    print(f"  Query: {query_image.shape}")
    print(f"  Reference: {reference_image.shape}")
    print(f"  Mask: {reference_mask.shape}")
    
    # Test 1: Memory Bank
    print(f"\n1. Testing Memory Bank...")
    memory_bank = TaskMemoryBank(device='cpu')
    
    # Store task embedding
    with torch.no_grad():
        task_embedding = model.encode_task(reference_image, reference_mask)
    
    memory_bank.store_embedding('liver', task_embedding, 'AMOS')
    
    # Retrieve embedding
    retrieved = memory_bank.retrieve_embedding('liver')
    assert retrieved is not None
    assert torch.allclose(task_embedding, retrieved, atol=1e-6)
    
    stats = memory_bank.get_stats()
    print(f"  Memory bank stats: {stats}")
    print("  ✅ Memory bank test passed")
    
    # Test 2: Sliding Window
    print(f"\n2. Testing Sliding Window...")
    sliding_window = SlidingWindowInference(
        patch_size=(8, 16, 16), overlap=0.5, batch_size=1
    )
    
    # Test patch extraction
    test_volume = torch.randn(1, 32, 64, 64)
    patches, coordinates = sliding_window.extract_patches(test_volume)
    print(f"  Extracted {len(patches)} patches")
    print(f"  Patch shape: {patches[0].shape}")
    
    # Test patch combination
    fake_predictions = torch.randn(len(patches), 1, 8, 16, 16)
    combined = sliding_window.combine_patches(
        fake_predictions, coordinates, (1, 32, 64, 64)
    )
    print(f"  Combined shape: {combined.shape}")
    print("  ✅ Sliding window test passed")
    
    # Test 3: Multi-Class Inference
    print(f"\n3. Testing Multi-Class Inference...")
    multi_class = MultiClassInference(model, memory_bank)
    
    # Store multiple embeddings
    memory_bank.store_embedding('kidney', task_embedding * 0.8, 'AMOS')
    memory_bank.store_embedding('spleen', task_embedding * 1.2, 'AMOS')
    
    # Multi-class segmentation
    class_names = ['liver', 'kidney', 'spleen']
    results = multi_class.segment_multiple_classes(query_image, class_names)
    
    print(f"  Segmented {len(results)} classes")
    for class_name, prediction in results.items():
        print(f"    {class_name}: {prediction.shape}")
    print("  ✅ Multi-class inference test passed")
    
    # Test 4: Complete Inference Engine
    print(f"\n4. Testing Complete Inference Engine...")
    engine = IRISInferenceEngine(model, device='cpu')
    
    # One-shot inference
    result = engine.one_shot_inference(query_image, reference_image, reference_mask)
    print(f"  One-shot result keys: {list(result.keys())}")
    
    # Store embedding and test memory bank inference
    engine.memory_bank.store_embedding('test_organ', result['task_embedding'])
    memory_result = engine.memory_bank_inference(query_image, 'test_organ')
    print(f"  Memory bank result keys: {list(memory_result.keys())}")
    
    print("  ✅ Complete inference engine test passed")
    
    print(f"\n🎉 All inference strategy tests passed!")
    
    return True


if __name__ == "__main__":
    test_inference_strategies()
