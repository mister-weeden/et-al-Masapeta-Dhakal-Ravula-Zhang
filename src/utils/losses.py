"""
Loss Functions for IRIS Framework

This module implements the loss functions used in IRIS training:
- Dice Loss: For segmentation quality
- CrossEntropy Loss: For classification
- Combined Loss: Weighted combination of both
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    
    Computes the Dice coefficient loss between predictions and targets.
    Handles both binary and multi-class segmentation.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-5)
        reduction (str): Reduction method ('mean', 'sum', 'none') (default: 'mean')
        ignore_index (int): Index to ignore in loss computation (default: None)
    """
    
    def __init__(self, smooth=1e-5, reduction='mean', ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, predictions, targets):
        """
        Compute Dice loss.
        
        Args:
            predictions: Model predictions (B, C, D, H, W) - logits or probabilities
            targets: Ground truth masks (B, C, D, H, W) or (B, D, H, W)
        
        Returns:
            Dice loss value
        """
        # Handle different input formats
        if predictions.dim() == 5 and targets.dim() == 4:
            # predictions: (B, C, D, H, W), targets: (B, D, H, W)
            targets = targets.unsqueeze(1)  # (B, 1, D, H, W)
        
        # Apply sigmoid/softmax to predictions if they are logits
        if predictions.dim() == 5 and predictions.shape[1] > 1:
            # Multi-class: apply softmax
            predictions = F.softmax(predictions, dim=1)
        else:
            # Binary: apply sigmoid
            predictions = torch.sigmoid(predictions)
        
        # Ensure same shape
        if predictions.shape != targets.shape:
            if targets.shape[1] == 1 and predictions.shape[1] > 1:
                # Convert targets to one-hot encoding
                targets = F.one_hot(targets.long().squeeze(1), num_classes=predictions.shape[1])
                targets = targets.permute(0, 4, 1, 2, 3).float()
        
        # Handle ignore_index
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            predictions = predictions * mask
            targets = targets * mask
        
        # Compute Dice coefficient for each class
        dice_scores = []
        num_classes = predictions.shape[1]
        
        for c in range(num_classes):
            pred_c = predictions[:, c]
            target_c = targets[:, c] if targets.shape[1] > 1 else targets[:, 0]
            
            # Flatten spatial dimensions
            pred_flat = pred_c.view(pred_c.shape[0], -1)
            target_flat = target_c.view(target_c.shape[0], -1)
            
            # Compute Dice coefficient
            intersection = (pred_flat * target_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Stack and compute loss
        dice_scores = torch.stack(dice_scores, dim=1)  # (B, C)
        dice_loss = 1.0 - dice_scores
        
        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function using Dice Loss and CrossEntropy Loss.
    
    Args:
        dice_weight (float): Weight for Dice loss (default: 0.5)
        ce_weight (float): Weight for CrossEntropy loss (default: 0.5)
        smooth (float): Smoothing factor for Dice loss (default: 1e-5)
        ignore_index (int): Index to ignore in loss computation (default: None)
    """
    
    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-5, ignore_index=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        
        if ignore_index is not None:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
            
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions (B, C, D, H, W)
            targets: Ground truth masks (B, D, H, W) or (B, C, D, H, W)
        
        Returns:
            Combined loss value
        """
        # Compute Dice loss
        dice_loss = self.dice_loss(predictions, targets)
        
        # Compute CrossEntropy/BCE loss
        if predictions.shape[1] == 1:
            # Binary segmentation: use BCE
            if targets.dim() == 5:
                targets = targets.squeeze(1)  # Remove channel dimension
            ce_loss = self.bce_loss(predictions.squeeze(1), targets.float())
        else:
            # Multi-class segmentation: use CrossEntropy
            if targets.dim() == 5:
                targets = targets.squeeze(1)  # Remove channel dimension
            ce_loss = self.ce_loss(predictions, targets.long())
        
        # Combine losses
        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        
        return total_loss, dice_loss, ce_loss


def dice_score(predictions, targets, smooth=1e-5):
    """
    Compute Dice score metric (not loss).
    
    Args:
        predictions: Model predictions (B, C, D, H, W) - probabilities
        targets: Ground truth masks (B, C, D, H, W) or (B, D, H, W)
        smooth: Smoothing factor
    
    Returns:
        Dice score (higher is better)
    """
    # Handle different input formats
    if predictions.dim() == 5 and targets.dim() == 4:
        targets = targets.unsqueeze(1)
    
    # Apply threshold to predictions if they are probabilities
    if predictions.max() <= 1.0:
        predictions = (predictions > 0.5).float()
    else:
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()
    
    # Compute Dice coefficient
    dice_scores = []
    num_classes = predictions.shape[1]
    
    for c in range(num_classes):
        pred_c = predictions[:, c]
        target_c = targets[:, c] if targets.shape[1] > 1 else targets[:, 0]
        
        # Flatten spatial dimensions
        pred_flat = pred_c.view(pred_c.shape[0], -1)
        target_flat = target_c.view(target_c.shape[0], -1)
        
        # Compute Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    
    dice_scores = torch.stack(dice_scores, dim=1)  # (B, C)
    return dice_scores.mean()


def test_loss_functions():
    """Test loss functions with synthetic data."""
    print("Testing Loss Functions...")
    
    # Test parameters
    batch_size = 2
    num_classes = 1
    depth, height, width = 16, 32, 32
    
    # Create synthetic data
    predictions = torch.randn(batch_size, num_classes, depth, height, width)
    targets = torch.randint(0, 2, (batch_size, depth, height, width)).float()
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Test Dice Loss
    print("\n1. Testing Dice Loss...")
    dice_loss_fn = DiceLoss()
    dice_loss = dice_loss_fn(predictions, targets)
    print(f"   Dice loss: {dice_loss.item():.4f}")
    assert dice_loss.item() >= 0, "Dice loss should be non-negative"
    
    # Test Combined Loss
    print("\n2. Testing Combined Loss...")
    combined_loss_fn = CombinedLoss()
    total_loss, dice_component, ce_component = combined_loss_fn(predictions, targets)
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Dice component: {dice_component.item():.4f}")
    print(f"   CE component: {ce_component.item():.4f}")
    
    # Test Dice Score Metric
    print("\n3. Testing Dice Score Metric...")
    dice_metric = dice_score(torch.sigmoid(predictions), targets)
    print(f"   Dice score: {dice_metric.item():.4f}")
    assert 0 <= dice_metric.item() <= 1, "Dice score should be between 0 and 1"
    
    # Test Multi-class
    print("\n4. Testing Multi-class...")
    multi_predictions = torch.randn(batch_size, 3, depth, height, width)
    multi_targets = torch.randint(0, 3, (batch_size, depth, height, width))
    
    multi_loss_fn = CombinedLoss()
    multi_total_loss, multi_dice, multi_ce = multi_loss_fn(multi_predictions, multi_targets)
    print(f"   Multi-class total loss: {multi_total_loss.item():.4f}")
    
    # Test gradient flow
    print("\n5. Testing gradient flow...")
    predictions.requires_grad_(True)
    loss = combined_loss_fn(predictions, targets)[0]
    loss.backward()
    assert predictions.grad is not None, "Gradients should flow to predictions"
    print("   ✓ Gradients flow correctly")
    
    print("\n✓ All loss function tests passed!")


if __name__ == "__main__":
    test_loss_functions()
