"""
Episodic Trainer for IRIS Framework

This module implements the episodic training loop for in-context learning:
1. Sample reference (xs, ys) and query (xq, yq) from same dataset
2. Extract task embedding: T = model.encode_task(xs, ys)
3. Predict: y_pred = model.decode(xq, T)
4. Loss: Dice + CrossEntropy
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Try to import tensorboard, fallback to dummy if not available
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: tensorboard not available, using dummy writer")
    TENSORBOARD_AVAILABLE = False
    
    class DummySummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
    
    SummaryWriter = DummySummaryWriter

import os
import time
from typing import Dict, Optional, Tuple
import numpy as np
from collections import defaultdict
import json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.iris_model import IRISModel
from utils.losses import CombinedLoss, dice_score
from data.episodic_loader import EpisodicDataLoader, DatasetRegistry


class EpisodicTrainer:
    """
    Episodic trainer for IRIS model.
    
    Implements the training loop for in-context learning where the model
    learns to segment using reference examples without fine-tuning.
    
    Args:
        model: IRIS model to train
        train_loader: Episodic data loader for training
        val_loader: Episodic data loader for validation (optional)
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler (optional)
        loss_fn: Loss function (default: CombinedLoss)
        device: Device to train on (default: 'cuda')
        log_dir: Directory for tensorboard logs (default: 'runs')
    """
    
    def __init__(self, model: IRISModel, train_loader: EpisodicDataLoader,
                 val_loader: Optional[EpisodicDataLoader] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 loss_fn: Optional[nn.Module] = None,
                 device: str = 'cuda',
                 log_dir: str = 'runs'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_dir = log_dir
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-5
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        self.scheduler = scheduler
        
        # Setup loss function
        if loss_fn is None:
            self.loss_fn = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
        else:
            self.loss_fn = loss_fn
        
        # Setup logging
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_dice = 0.0
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        print(f"EpisodicTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Optimizer: {type(self.optimizer).__name__}")
        print(f"  Loss function: {type(self.loss_fn).__name__}")
    
    def episodic_training_step(self, episode) -> Dict[str, float]:
        """
        Perform a single episodic training step.
        
        Args:
            episode: EpisodicSample containing reference and query data
        
        Returns:
            Dictionary of metrics for this step
        """
        # Move data to device
        reference_image = episode.reference_image.unsqueeze(0).to(self.device)  # (1, C, D, H, W)
        reference_mask = episode.reference_mask.unsqueeze(0).to(self.device)    # (1, C, D, H, W)
        query_image = episode.query_image.unsqueeze(0).to(self.device)          # (1, C, D, H, W)
        query_mask = episode.query_mask.unsqueeze(0).to(self.device)            # (1, C, D, H, W)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        # 1. Extract task embedding from reference
        task_embedding = self.model.encode_task(reference_image, reference_mask)
        
        # 2. Segment query image using task embedding
        prediction = self.model.segment_with_task(query_image, task_embedding)
        
        # 3. Compute loss
        total_loss, dice_loss, ce_loss = self.loss_fn(prediction, query_mask.squeeze(1))
        
        # 4. Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            dice_metric = dice_score(torch.sigmoid(prediction), query_mask)
            
            # Task embedding statistics
            task_emb_mean = task_embedding.mean().item()
            task_emb_std = task_embedding.std().item()
        
        metrics = {
            'total_loss': total_loss.item(),
            'dice_loss': dice_loss.item(),
            'ce_loss': ce_loss.item(),
            'dice_score': dice_metric.item(),
            'task_emb_mean': task_emb_mean,
            'task_emb_std': task_emb_std,
            'class_name': episode.class_name,
            'dataset_name': episode.dataset_name
        }
        
        return metrics
    
    def validation_step(self, episode) -> Dict[str, float]:
        """
        Perform a single validation step.
        
        Args:
            episode: EpisodicSample for validation
        
        Returns:
            Dictionary of validation metrics
        """
        with torch.no_grad():
            # Move data to device
            reference_image = episode.reference_image.unsqueeze(0).to(self.device)
            reference_mask = episode.reference_mask.unsqueeze(0).to(self.device)
            query_image = episode.query_image.unsqueeze(0).to(self.device)
            query_mask = episode.query_mask.unsqueeze(0).to(self.device)
            
            # Forward pass
            task_embedding = self.model.encode_task(reference_image, reference_mask)
            prediction = self.model.segment_with_task(query_image, task_embedding)
            
            # Compute loss and metrics
            total_loss, dice_loss, ce_loss = self.loss_fn(prediction, query_mask.squeeze(1))
            dice_metric = dice_score(torch.sigmoid(prediction), query_mask)
            
            metrics = {
                'val_total_loss': total_loss.item(),
                'val_dice_loss': dice_loss.item(),
                'val_ce_loss': ce_loss.item(),
                'val_dice_score': dice_metric.item(),
                'class_name': episode.class_name,
                'dataset_name': episode.dataset_name
            }
        
        return metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = defaultdict(list)
        class_metrics = defaultdict(list)
        
        start_time = time.time()
        
        for step, episode in enumerate(self.train_loader):
            # Training step
            step_metrics = self.episodic_training_step(episode)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in ['class_name', 'dataset_name']:
                    epoch_metrics[key].append(value)
            
            # Track per-class metrics
            class_name = step_metrics['class_name']
            class_metrics[class_name].append(step_metrics['dice_score'])
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                for key, value in step_metrics.items():
                    if key not in ['class_name', 'dataset_name']:
                        self.writer.add_scalar(f'train/{key}', value, self.global_step)
            
            # Print progress
            if step % 50 == 0:
                print(f"  Step {step:4d}: Loss={step_metrics['total_loss']:.4f}, "
                      f"Dice={step_metrics['dice_score']:.4f}, Class={class_name}")
            
            self.global_step += 1
        
        # Compute epoch averages
        epoch_avg = {}
        for key, values in epoch_metrics.items():
            epoch_avg[key] = np.mean(values)
        
        # Compute per-class averages
        class_avg = {}
        for class_name, dice_scores in class_metrics.items():
            class_avg[class_name] = np.mean(dice_scores)
        
        epoch_time = time.time() - start_time
        epoch_avg['epoch_time'] = epoch_time
        
        print(f"Epoch {self.current_epoch} Training Summary:")
        print(f"  Total Loss: {epoch_avg['total_loss']:.4f}")
        print(f"  Dice Score: {epoch_avg['dice_score']:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Classes trained: {len(class_avg)}")
        
        return epoch_avg, class_avg
    
    def validate_epoch(self) -> Optional[Dict[str, float]]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        
        epoch_metrics = defaultdict(list)
        class_metrics = defaultdict(list)
        
        start_time = time.time()
        
        for step, episode in enumerate(self.val_loader):
            # Validation step
            step_metrics = self.validation_step(episode)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in ['class_name', 'dataset_name']:
                    epoch_metrics[key].append(value)
            
            # Track per-class metrics
            class_name = step_metrics['class_name']
            class_metrics[class_name].append(step_metrics['val_dice_score'])
        
        # Compute epoch averages
        epoch_avg = {}
        for key, values in epoch_metrics.items():
            epoch_avg[key] = np.mean(values)
        
        # Compute per-class averages
        class_avg = {}
        for class_name, dice_scores in class_metrics.items():
            class_avg[class_name] = np.mean(dice_scores)
        
        epoch_time = time.time() - start_time
        epoch_avg['epoch_time'] = epoch_time
        
        print(f"Epoch {self.current_epoch} Validation Summary:")
        print(f"  Val Dice Score: {epoch_avg['val_dice_score']:.4f}")
        print(f"  Val Total Loss: {epoch_avg['val_total_loss']:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Log to tensorboard
        for key, value in epoch_avg.items():
            if key != 'epoch_time':
                self.writer.add_scalar(f'val/{key}', value, self.current_epoch)
        
        return epoch_avg, class_avg
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_dice': self.best_val_dice,
            'model_config': self.model.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_dice = checkpoint['best_val_dice']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int, save_dir: str = 'checkpoints'):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Checkpoints will be saved to: {save_dir}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_metrics, train_class_metrics = self.train_epoch()
            
            # Validation
            val_metrics = None
            if self.val_loader is not None:
                val_metrics, val_class_metrics = self.validate_epoch()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics is not None:
                        self.scheduler.step(val_metrics['val_dice_score'])
                    else:
                        self.scheduler.step(train_metrics['dice_score'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            is_best = False
            
            if val_metrics is not None:
                current_dice = val_metrics['val_dice_score']
                if current_dice > self.best_val_dice:
                    self.best_val_dice = current_dice
                    is_best = True
            
            self.save_checkpoint(checkpoint_path, is_best)
            
            # Log epoch metrics
            self.writer.add_scalar('epoch/train_dice', train_metrics['dice_score'], epoch)
            if val_metrics is not None:
                self.writer.add_scalar('epoch/val_dice', val_metrics['val_dice_score'], epoch)
        
        print(f"\nTraining completed!")
        print(f"Best validation Dice score: {self.best_val_dice:.4f}")
        
        self.writer.close()


def test_episodic_trainer():
    """Test episodic trainer with synthetic data."""
    print("Testing Episodic Trainer...")
    
    # Import required modules
    from data.episodic_loader import create_amos_registry
    
    # Create synthetic setup
    model = IRISModel(
        in_channels=1,
        base_channels=8,   # Very small for testing
        embed_dim=32,      # Very small for testing
        num_tokens=3,      # Very small for testing
        num_classes=1
    )
    
    # Create data loaders
    registry = create_amos_registry()
    train_loader = EpisodicDataLoader(
        registry=registry,
        episode_size=2,
        max_episodes_per_epoch=20,  # Very small for testing
        spatial_size=(16, 32, 32),  # Very small for testing
        augment=False
    )
    
    val_loader = EpisodicDataLoader(
        registry=registry,
        episode_size=2,
        max_episodes_per_epoch=10,  # Very small for testing
        spatial_size=(16, 32, 32),
        augment=False
    )
    
    # Create trainer
    trainer = EpisodicTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu',  # Use CPU for testing
        log_dir='test_runs'
    )
    
    print("Testing single training step...")
    
    # Test single episode
    episode = next(iter(train_loader))
    metrics = trainer.episodic_training_step(episode)
    
    print("Training step metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test validation step
    val_metrics = trainer.validation_step(episode)
    print("\nValidation step metrics:")
    for key, value in val_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✓ Episodic trainer test completed!")


if __name__ == "__main__":
    test_episodic_trainer()
