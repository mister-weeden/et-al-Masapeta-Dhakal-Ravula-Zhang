"""
Episodic Data Loader for IRIS Framework

This module implements episodic sampling for in-context learning:
- Sample reference and query pairs from same anatomical class
- Handle multiple datasets (AMOS, BCV, etc.)
- Ensure reference/query are from different patients
- Support binary decomposition for multi-class datasets
"""

import torch
import torch.utils.data as data
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import os
import json


class EpisodicSample:
    """Container for an episodic training sample."""
    
    def __init__(self, reference_image, reference_mask, query_image, query_mask, 
                 class_name, dataset_name):
        self.reference_image = reference_image
        self.reference_mask = reference_mask
        self.query_image = query_image
        self.query_mask = query_mask
        self.class_name = class_name
        self.dataset_name = dataset_name


class DatasetRegistry:
    """Registry for managing multiple medical imaging datasets."""
    
    def __init__(self):
        self.datasets = {}
        self.class_mappings = {}
        
    def register_dataset(self, name: str, data_path: str, class_mapping: Dict[str, int]):
        """
        Register a dataset with its class mapping.
        
        Args:
            name: Dataset name (e.g., 'AMOS', 'BCV')
            data_path: Path to dataset files
            class_mapping: Mapping from class names to label values
        """
        self.datasets[name] = {
            'path': data_path,
            'class_mapping': class_mapping,
            'samples': []
        }
        self.class_mappings[name] = class_mapping
    
    def add_sample(self, dataset_name: str, image_path: str, mask_path: str, 
                   patient_id: str, available_classes: List[str]):
        """Add a sample to the dataset registry."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not registered")
        
        sample = {
            'image_path': image_path,
            'mask_path': mask_path,
            'patient_id': patient_id,
            'available_classes': available_classes
        }
        self.datasets[dataset_name]['samples'].append(sample)
    
    def get_samples_by_class(self, dataset_name: str, class_name: str) -> List[Dict]:
        """Get all samples containing a specific class."""
        if dataset_name not in self.datasets:
            return []
        
        samples = []
        for sample in self.datasets[dataset_name]['samples']:
            if class_name in sample['available_classes']:
                samples.append(sample)
        return samples
    
    def get_all_classes(self) -> Dict[str, List[str]]:
        """Get all available classes for each dataset."""
        all_classes = {}
        for dataset_name, dataset_info in self.datasets.items():
            all_classes[dataset_name] = list(dataset_info['class_mapping'].keys())
        return all_classes


class EpisodicDataLoader:
    """
    Episodic data loader for in-context learning.
    
    Samples reference and query pairs from the same anatomical class
    but different patients to enable in-context learning.
    
    Args:
        registry: DatasetRegistry containing all datasets
        episode_size: Number of samples per episode (default: 2 - reference + query)
        max_episodes_per_epoch: Maximum episodes per epoch (default: 1000)
        spatial_size: Target spatial size for images (default: (64, 128, 128))
        augment: Whether to apply data augmentation (default: True)
    """
    
    def __init__(self, registry: DatasetRegistry, episode_size: int = 2,
                 max_episodes_per_epoch: int = 1000, 
                 spatial_size: Tuple[int, int, int] = (64, 128, 128),
                 augment: bool = True):
        self.registry = registry
        self.episode_size = episode_size
        self.max_episodes_per_epoch = max_episodes_per_epoch
        self.spatial_size = spatial_size
        self.augment = augment
        
        # Build class index for efficient sampling
        self._build_class_index()
        
    def _build_class_index(self):
        """Build index of samples by class for efficient sampling."""
        self.class_index = defaultdict(list)
        
        for dataset_name, dataset_info in self.registry.datasets.items():
            for sample in dataset_info['samples']:
                for class_name in sample['available_classes']:
                    self.class_index[f"{dataset_name}:{class_name}"].append({
                        'dataset': dataset_name,
                        'sample': sample,
                        'class': class_name
                    })
        
        # Filter out classes with insufficient samples
        self.valid_classes = []
        for class_key, samples in self.class_index.items():
            if len(samples) >= self.episode_size:
                self.valid_classes.append(class_key)
        
        print(f"EpisodicDataLoader: {len(self.valid_classes)} valid classes for episodic sampling")
    
    def sample_episode(self) -> EpisodicSample:
        """
        Sample a single episode (reference + query pair).
        
        Returns:
            EpisodicSample containing reference and query data
        """
        # Randomly select a class
        class_key = random.choice(self.valid_classes)
        available_samples = self.class_index[class_key]
        
        # Sample reference and query from different patients
        sampled_indices = random.sample(range(len(available_samples)), self.episode_size)
        reference_info = available_samples[sampled_indices[0]]
        query_info = available_samples[sampled_indices[1]]
        
        # Ensure different patients
        max_attempts = 10
        attempts = 0
        while (reference_info['sample']['patient_id'] == query_info['sample']['patient_id'] 
               and attempts < max_attempts):
            sampled_indices = random.sample(range(len(available_samples)), self.episode_size)
            reference_info = available_samples[sampled_indices[0]]
            query_info = available_samples[sampled_indices[1]]
            attempts += 1
        
        # Load reference data
        ref_image = self._load_image(reference_info['dataset'], reference_info['sample']['image_path'])
        ref_mask = self._load_mask(reference_info['dataset'], reference_info['sample']['mask_path'], 
                                  reference_info['class'])
        
        # Load query data
        query_image = self._load_image(query_info['dataset'], query_info['sample']['image_path'])
        query_mask = self._load_mask(query_info['dataset'], query_info['sample']['mask_path'], 
                                    query_info['class'])
        
        # Apply preprocessing
        ref_image, ref_mask = self._preprocess(ref_image, ref_mask)
        query_image, query_mask = self._preprocess(query_image, query_mask)
        
        return EpisodicSample(
            reference_image=ref_image,
            reference_mask=ref_mask,
            query_image=query_image,
            query_mask=query_mask,
            class_name=reference_info['class'],
            dataset_name=reference_info['dataset']
        )
    
    def _load_image(self, dataset_name: str, image_path: str) -> torch.Tensor:
        """Load and preprocess medical image."""
        # This is a placeholder - in real implementation, would use SimpleITK/nibabel
        # For now, return synthetic data matching expected format
        return torch.randn(1, *self.spatial_size)  # (C, D, H, W)
    
    def _load_mask(self, dataset_name: str, mask_path: str, class_name: str) -> torch.Tensor:
        """Load and extract binary mask for specific class."""
        # This is a placeholder - in real implementation, would:
        # 1. Load multi-class mask
        # 2. Extract binary mask for specific class
        # 3. Handle class mapping
        
        # Create synthetic binary mask
        mask = torch.zeros(1, *self.spatial_size)
        
        # Add some structure to the mask
        d, h, w = self.spatial_size
        center_d, center_h, center_w = d//2, h//2, w//2
        radius = min(d, h, w) // 6
        
        for i in range(d):
            for j in range(h):
                for k in range(w):
                    dist = ((i - center_d)**2 + (j - center_h)**2 + (k - center_w)**2)**0.5
                    if dist < radius:
                        mask[0, i, j, k] = 1.0
        
        return mask
    
    def _preprocess(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply preprocessing to image and mask."""
        # Normalize image
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        # Ensure mask is binary
        mask = (mask > 0.5).float()
        
        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        
        return image, mask
    
    def _apply_augmentation(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation."""
        # Simple augmentation: random flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-1])  # Flip width
            mask = torch.flip(mask, dims=[-1])
        
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-2])  # Flip height
            mask = torch.flip(mask, dims=[-2])
        
        return image, mask
    
    def __iter__(self):
        """Iterator for episodic sampling."""
        for _ in range(self.max_episodes_per_epoch):
            yield self.sample_episode()
    
    def __len__(self):
        return self.max_episodes_per_epoch


def create_amos_registry() -> DatasetRegistry:
    """Create a registry with AMOS22 dataset configuration."""
    registry = DatasetRegistry()
    
    # AMOS22 class mapping (15 anatomical structures)
    amos_classes = {
        'spleen': 1,
        'right_kidney': 2,
        'left_kidney': 3,
        'gallbladder': 4,
        'esophagus': 5,
        'liver': 6,
        'stomach': 7,
        'aorta': 8,
        'inferior_vena_cava': 9,
        'portal_vein_splenic_vein': 10,
        'pancreas': 11,
        'right_adrenal_gland': 12,
        'left_adrenal_gland': 13,
        'duodenum': 14,
        'bladder': 15
    }
    
    registry.register_dataset('AMOS', '/path/to/amos22', amos_classes)
    
    # Add synthetic samples for testing
    for i in range(100):  # 100 synthetic patients
        patient_id = f"AMOS_{i:03d}"
        # Each patient has random subset of organs
        available_classes = random.sample(list(amos_classes.keys()), 
                                        random.randint(5, 10))
        
        registry.add_sample(
            dataset_name='AMOS',
            image_path=f'/path/to/amos22/images/{patient_id}.nii.gz',
            mask_path=f'/path/to/amos22/masks/{patient_id}.nii.gz',
            patient_id=patient_id,
            available_classes=available_classes
        )
    
    return registry


def test_episodic_loader():
    """Test episodic data loader functionality."""
    print("Testing Episodic Data Loader...")
    
    # Create test registry
    registry = create_amos_registry()
    
    print(f"Registry contains {len(registry.datasets)} datasets")
    for dataset_name, dataset_info in registry.datasets.items():
        print(f"  {dataset_name}: {len(dataset_info['samples'])} samples")
    
    # Create episodic loader
    loader = EpisodicDataLoader(
        registry=registry,
        episode_size=2,
        max_episodes_per_epoch=10,  # Small for testing
        spatial_size=(32, 64, 64),  # Smaller for testing
        augment=True
    )
    
    print(f"Loader has {len(loader.valid_classes)} valid classes")
    
    # Test episode sampling
    print("\nTesting episode sampling...")
    episode_count = 0
    class_counts = defaultdict(int)
    
    for episode in loader:
        episode_count += 1
        class_counts[episode.class_name] += 1
        
        print(f"Episode {episode_count}:")
        print(f"  Class: {episode.class_name}")
        print(f"  Dataset: {episode.dataset_name}")
        print(f"  Reference image: {episode.reference_image.shape}")
        print(f"  Reference mask: {episode.reference_mask.shape}, coverage: {episode.reference_mask.mean():.3f}")
        print(f"  Query image: {episode.query_image.shape}")
        print(f"  Query mask: {episode.query_mask.shape}, coverage: {episode.query_mask.mean():.3f}")
        
        # Verify shapes
        assert episode.reference_image.shape == (1, 32, 64, 64)
        assert episode.reference_mask.shape == (1, 32, 64, 64)
        assert episode.query_image.shape == (1, 32, 64, 64)
        assert episode.query_mask.shape == (1, 32, 64, 64)
        
        # Verify masks are binary
        assert episode.reference_mask.min() >= 0 and episode.reference_mask.max() <= 1
        assert episode.query_mask.min() >= 0 and episode.query_mask.max() <= 1
        
        if episode_count >= 5:  # Test first 5 episodes
            break
    
    print(f"\nClass distribution in {episode_count} episodes:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    print("\n✓ All episodic loader tests passed!")


if __name__ == "__main__":
    test_episodic_loader()
