# IRIS: Universal Medical Image Segmentation via In-Context Learning

## Overview

This repository provides a minimal implementation to reproduce and validate the core claims of the IRIS framework from "Show and Segment: Universal Medical Image Segmentation via In-Context Learning" (Gao et al., 2025).

## Core Hypothesis & Claims

### Primary Hypothesis
Medical image segmentation can be performed on **unseen anatomical structures** without fine-tuning by using reference image-label pairs as context, similar to how large language models perform in-context learning.

### Key Claims to Validate
1. **In-context adaptation**: Model can segment novel anatomical structures using only a single reference example
2. **Superior generalization**: Achieves 82-86% Dice score on out-of-distribution datasets
3. **Novel class performance**: Achieves 28-69% Dice score on completely unseen anatomical structures
4. **Efficiency**: Single forward pass for multi-class segmentation (unlike existing methods)

### Key Components

1. **Task Encoding Module**
   - Foreground feature extraction at high resolution
   - Contextual encoding using learnable query tokens
   - Produces compact task embedding T ∈ R^(m+1)×C

2. **Image Encoder**
   - 3D UNet with 4 downsampling stages
   - Base channels: 32
   - Residual connections

3. **Mask Decoder**
   - Query-based architecture with cross-attention
   - Integrates task embeddings with query features

## Prerequisites

### System Requirements
- GPU: NVIDIA A100 (40GB) or similar
- RAM: 64GB minimum
- CUDA: 11.7+
- Python: 3.9+

### Core Libraries
```bash
# PyTorch and CUDA
pytorch==2.0.0
torchvision==0.15.0
pytorch-cuda==11.7

# Medical Imaging
SimpleITK==2.2.1
nibabel==5.1.0
monai==1.2.0

# Core Dependencies
numpy==1.24.0
scipy==1.10.0
scikit-learn==1.2.0
pandas==1.5.0

# Training Tools
tensorboard==2.12.0
wandb==0.14.0
tqdm==4.65.0
pyyaml==6.0
hydra-core==1.3.0
```

## Datasets

### Training Datasets (12 total)
1. **AMOS** (Abdominal Multi-Organ): CT/MRI, 15 organs
2. **BCV** (Beyond Cranial Vault): CT, 13 organs
3. **CHAOS**: MRI, 4 organs
4. **KiTS19**: CT, kidney & tumor
5. **LiTS**: CT, liver & tumor
6. **M&Ms**: Cardiac MRI, 3 structures
7. **Brain Tumor**: MRI, 3 tissue types
8. **AutoPET**: PET/CT, lesions
9. **StructSeg H&N**: CT, 22 organs
10. **StructSeg Tho**: CT, 6 organs
11. **CSI-Wat**: Spine MRI, intervertebral discs
12. Additional datasets as per paper

### Test Datasets (Held-out)
- **ACDC**: Cardiac MRI (generalization)
- **SegTHOR**: Thoracic CT (generalization)
- **MSD Pancreas**: Pancreas tumor (novel class)
- **Pelvic1K**: Pelvic bones (novel class)

### Data Access
- AMOS: https://amos22.grand-challenge.org/
- Medical Segmentation Decathlon: http://medicaldecathlon.com/
- KiTS19: https://kits19.grand-challenge.org/

## Minimal Implementation Strategy

### 1. Core Task Encoding Module
```python
class TaskEncodingModule(nn.Module):
    """Extract task-specific features from reference examples"""
    def __init__(self, in_channels=512, embed_dim=512, num_tokens=10):
        # Foreground encoding: high-res feature extraction
        # Context encoding: learnable query tokens + attention
```

### 2. Episodic Training
```python
# Sample reference and query pairs from same dataset
# Extract task embedding from reference
# Use embedding to guide query segmentation
```

### 3. Key Experiments to Validate

#### Experiment 1: In-Context Adaptation
- Test on AMOS dataset with single reference examples
- Compare one-shot performance vs. task-specific models
- **Success Metric**: >70% Dice score without fine-tuning

#### Experiment 2: Novel Class Segmentation
- Train on 11 datasets, test on MSD Pancreas (tumor)
- Use single reference example at test time
- **Success Metric**: >25% Dice score (paper claims 28.28%)

#### Experiment 3: Generalization
- Train on structured datasets, test on ACDC/SegTHOR
- **Success Metric**: >80% Dice score (paper claims 82-86%)

## Critical Implementation Details

### Missing from Paper (Assumptions Required)
1. **Attention mechanism**: Assumed 2-layer transformer with 8 heads
2. **PixelShuffle3D**: Custom implementation for 3D volumes
3. **Memory bank**: EMA update with momentum=0.999
4. **Inference overlap**: 50% sliding window overlap
5. **Training schedule**: 80K iterations, Lamb optimizer

### Minimum Viable Implementation
1. Task encoding module with foreground/context paths
2. Simple 3D UNet encoder-decoder
3. Episodic training loop
4. Basic inference with single reference
5. Dice score evaluation

## Quick Start

```bash
# 1. Setup environment
conda create -n iris python=3.9
conda activate iris
pip install -r requirements.txt

# 2. Download minimal dataset (e.g., AMOS subset)
python scripts/download_data.py --dataset amos --subset mini

# 3. Train minimal model
python train_minimal.py --config configs/minimal.yaml

# 4. Evaluate in-context learning
python evaluate_incontext.py --checkpoint runs/minimal/best.pth
```

## Expected Results

### Minimal Implementation
- In-distribution: 70-80% Dice (vs. paper's 89.56%)
- Novel classes: 20-30% Dice (vs. paper's 28-69%)
- Training time: 12-24 hours on single GPU

### Full Implementation
- Requires all architectural details from paper
- Multi-GPU training recommended
- 48-72 hour training time

## Key Questions to Investigate

1. **Does foreground feature extraction actually help?**
   - Ablation: Remove high-resolution processing
   - Expected impact: 15-20% performance drop

2. **Is the task encoding reusable across queries?**
   - Test: Encode once, apply to 100 queries
   - Expected: Consistent performance

3. **How sensitive is performance to reference quality?**
   - Test: Random vs. curated reference selection
   - Expected: 10-15% performance variation

## Citation

```bibtex
@article{gao2025show,
  title={Show and Segment: Universal Medical Image Segmentation via In-Context Learning},
  author={Gao, Yunhe and others},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This implementation is for research purposes only. Please refer to individual dataset licenses for usage restrictions.