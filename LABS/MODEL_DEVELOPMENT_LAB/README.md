# Distributed Training Lab: Alzheimer's Disease Classification

## Overview
This lab demonstrates distributed training using **Ray Train** for an Alzheimer's disease classification model. The project compares single GPU training (baseline) with distributed training across multiple workers to understand the concepts and benefits of distributed deep learning.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Approaches](#training-approaches)
- [Results](#results)
- [Key Learnings](#key-learnings)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Project Description

### Objective
Train a Convolutional Neural Network (CNN) to classify brain MRI scans into four categories of Alzheimer's disease severity:
- **NonDemented**: Healthy individuals
- **VeryMildDemented**: Very mild cognitive impairment
- **MildDemented**: Mild Alzheimer's disease
- **ModerateDemented**: Moderate Alzheimer's disease

### Goals
1. Implement baseline training on a single GPU
2. Implement distributed training using Ray Train
3. Compare performance between single and distributed approaches
4. Understand distributed training concepts and infrastructure

---

## Dataset

### Dataset Statistics
- **Total Images**: 6,400
- **Training Set**: 5,121 images
- **Test Set**: 1,279 images
- **Image Format**: JPG/PNG
- **Image Size**: 128x128 pixels (resized)

### Class Distribution

#### Training Set:
| Class | Images | Percentage |
|-------|--------|------------|
| NonDemented | 2,560 | 50.0% |
| VeryMildDemented | 1,792 | 35.0% |
| MildDemented | 717 | 14.0% |
| ModerateDemented | 52 | 1.0% |

#### Test Set:
| Class | Images | Percentage |
|-------|--------|------------|
| NonDemented | 640 | 50.0% |
| VeryMildDemented | 448 | 35.0% |
| MildDemented | 179 | 14.0% |
| ModerateDemented | 12 | 1.0% |

### Data Preprocessing
- **Resize**: 128x128 pixels
- **Normalization**: Mean=[0.5], Std=[0.5]
- **Augmentation** (Training only):
  - Random horizontal flip (p=0.5)
  - Random rotation (±10 degrees)

---

## Model Architecture

### AlzheimerCNN
```
Input: 128x128x3 RGB images
│
├── Conv2D (32 filters, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
├── Conv2D (64 filters, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
├── Conv2D (128 filters, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
├── Conv2D (256 filters, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
│
├── Flatten
├── FC (512 units) + ReLU + Dropout(0.5)
├── FC (128 units) + ReLU + Dropout(0.3)
└── FC (4 units) - Output layer
```

### Model Statistics
- **Total Parameters**: ~16.7 million
- **Trainable Parameters**: ~16.7 million
- **Output**: 4 classes (Softmax activation)

---

## Training Approaches

### 1. Baseline Training (Single GPU)

**Configuration:**
- Hardware: 1x Tesla T4 GPU (15GB VRAM)
- Batch Size: 32
- Optimizer: Adam (lr=0.001)
- Loss Function: CrossEntropyLoss
- Epochs: 5

**Results:**
- Total Training Time: 56.39s
- Average Time per Epoch: 11.28s
- Final Test Accuracy: 35.03%

### 2. Distributed Training (Ray Train)

**Configuration:**
- Hardware: 2 CPU workers (simulated distributed training)
- Framework: Ray Train with PyTorch DDP backend
- Batch Size: 32 per worker
- Optimizer: Adam (lr=0.001)
- Loss Function: CrossEntropyLoss
- Epochs: 5
- Communication Backend: Gloo (CPU)

**Results:**
- Total Training Time: 1,109.50s
- Average Time per Epoch: 215.78s
- Final Test Accuracy: 51.56%

#### Epoch-by-Epoch Progress:
| Epoch | Train Loss | Train Acc | Test Acc | Time (s) |
|-------|------------|-----------|----------|----------|
| 1 | 1.307 | 46.86% | 35.00% | 221.66 |
| 2 | 0.964 | 52.21% | 35.00% | 216.32 |
| 3 | 0.938 | 55.13% | 56.41% | 215.81 |
| 4 | 0.905 | 54.43% | 35.31% | 209.36 |
| 5 | 0.900 | 54.78% | 51.56% | 215.62 |

---

## Results

### Performance Comparison

| Metric | Baseline (GPU) | Distributed (CPU) | Difference |
|--------|----------------|-------------------|------------|
| Avg Time/Epoch | 11.28s | 215.78s | 19.1x slower |
| Total Time | 56.39s | 1,109.50s | 19.7x slower |
| Final Test Acc | 35.03% | 51.56% | +16.53% |

### Why is Distributed Training Slower?

**Important Note:** In this lab, distributed training on CPU is slower because:
1. **Hardware Limitation**: CPUs are ~20x slower than GPUs for deep learning
2. **Single GPU Environment**: Google Colab provides only 1 GPU
3. **Communication Overhead**: Gradient synchronization between workers adds latency
4. **Simulation**: We simulated distributed training using CPU workers

### Real-World Distributed Training

In production environments with **multiple GPUs**:
- **2 GPUs**: ~1.8x speedup
- **4 GPUs**: ~3.5x speedup
- **8 GPUs**: ~7x speedup

**Linear Scaling** is achievable with:
- Proper hardware (multiple GPUs/nodes)
- Optimized communication (NCCL for NVIDIA GPUs)
- Efficient data loading and preprocessing

---

## Key Learnings

### 1. Distributed Training Concepts
- **Data Parallelism**: Split batches across multiple workers, each with a full model copy
- **Model Parallelism**: Split model layers across devices (not used in this lab)
- **Gradient Synchronization**: Workers compute gradients independently, then aggregate using all-reduce

### 2. Ray Train Benefits
- **Framework Agnostic**: Works with PyTorch, TensorFlow, XGBoost
- **Easy Scaling**: Change `num_workers` to scale horizontally
- **Fault Tolerance**: Automatic worker restart on failures
- **Unified API**: Single API for training, tuning, and serving

### 3. When to Use Distributed Training
**Use When:**
- Dataset too large for single GPU memory
- Model training takes hours/days
- Multiple GPUs available
- Need faster iteration cycles

**Don't Use When:**
- Small datasets (< 10k samples)
- Simple models (< 1M parameters)
- Only 1 GPU available
- Development/debugging phase

### 4. Best Practices
- Start with single GPU baseline
- Use data parallelism for large datasets
- Monitor GPU utilization
- Profile communication overhead
- Use mixed precision training (FP16)
- Increase batch size with more workers

---

## Requirements

### Hardware
- **Minimum**: 2 CPUs, 8GB RAM
- **Recommended**: 1+ GPUs, 16GB RAM, 50GB storage

### Software
```
Python: 3.8-3.11
Ray: 2.9.0+
PyTorch: 2.0.0+
torchvision: 0.15.0+
CUDA: 11.8+ (for GPU training)
```

### Python Packages
```
ray[train]==2.51.1
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
Pillow==10.0.0
tqdm==4.66.1
```

---

## Installation

### Option 1: Google Colab (Recommended for Lab)
```python
# Enable GPU: Runtime → Change runtime type → GPU (T4)

# Install packages
!pip install -U "ray[train]" torch torchvision matplotlib seaborn scikit-learn

# Verify installation
import ray
import torch
print(f"Ray version: {ray.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Option 2: Local Environment
```bash
# Create virtual environment
conda create -n ray-distributed python=3.10
conda activate ray-distributed

# Install packages
pip install -U "ray[train]"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn scikit-learn Pillow tqdm

# Verify
python -c "import ray; print(ray.__version__)"
```

---

## Usage

### 1. Setup Environment
```python
import ray
ray.init(num_cpus=2, num_gpus=1)
```

### 2. Load Dataset
```python
# Extract dataset
import zipfile
with zipfile.ZipFile('AlzheimerDataset4.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')

# Verify structure
import os
print(os.listdir('data/Alzheimer_s Dataset/train'))
# Output: ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
```

### 3. Train Baseline Model
```python
from torch.utils.data import DataLoader

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train
model = AlzheimerCNN(num_classes=4).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):
    # Training code here
    pass
```

### 4. Train with Ray Distributed
```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig

# Configure trainer
trainer = TorchTrainer(
    train_func_distributed,
    train_loop_config={
        "num_epochs": 5,
        "lr": 0.001,
        "batch_size": 32
    },
    scaling_config=ScalingConfig(
        num_workers=2,
        use_gpu=True,
        resources_per_worker={"CPU": 1, "GPU": 0.5}
    )
)

# Train
result = trainer.fit()
```

### 5. Compare Results
```python
import matplotlib.pyplot as plt

# Plot training curves
plt.plot(baseline_results['train_losses'], label='Baseline')
plt.plot(distributed_results['train_loss'], label='Distributed')
plt.legend()
plt.show()
```

---

## Project Structure
```
alzheimer-distributed-training/
│
├── data/                          # Dataset directory
│   └── Alzheimer_s Dataset/
│       ├── train/
│       │   ├── MildDemented/
│       │   ├── ModerateDemented/
│       │   ├── NonDemented/
│       │   └── VeryMildDemented/
│       └── test/
│           ├── MildDemented/
│           ├── ModerateDemented/
│           ├── NonDemented/
│           └── VeryMildDemented/
│
├── models/                        # Saved model checkpoints
│
├── results/                       # Training results and visualizations
│   ├── sample_images.png
│   └── training_comparison.png
│
├── checkpoints/                   # Ray checkpoints
│
├── README.md                      # This file
│
└── Alzheimers_Ray_Distributed_Training.ipynb  # Main notebook
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Solution: Reduce batch size
batch_size = 16  # Instead of 32
```

#### 2. Ray GPU Conflict
```
Error: Duplicate GPU detected
```
```python
# Solution: Use CPU for simulation
scaling_config = ScalingConfig(num_workers=2, use_gpu=False)
```

#### 3. File Not Found
```python
# Solution: Use absolute paths
import os
data_path = os.path.abspath('data/Alzheimer_s Dataset')
```

#### 4. Slow Training on CPU
- **Expected**: CPU is 20x slower than GPU
- **Solution**: Use GPU or reduce epochs for testing

---

## Future Improvements

### Short-term
- [ ] Implement mixed precision training (FP16)
- [ ] Add learning rate scheduling
- [ ] Use weighted loss for class imbalance
- [ ] Implement early stopping

### Long-term
- [ ] Try transfer learning (ResNet, EfficientNet)
- [ ] Multi-GPU training with proper hardware
- [ ] Hyperparameter tuning with Ray Tune
- [ ] Model compression and quantization
- [ ] Deploy model with Ray Serve

---

## References

### Frameworks & Tools
- [Ray Train Documentation](https://docs.ray.io/en/latest/train/train.html)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch DDP](https://pytorch.org/docs/stable/notes/ddp.html)

### Research Papers
- [Distributed Deep Learning](https://arxiv.org/abs/1404.5997)
- [Data Parallelism vs Model Parallelism](https://arxiv.org/abs/1811.06965)

### Datasets
- [OASIS Brain Dataset](https://www.oasis-brains.org/)
- [ADNI - Alzheimer's Disease Neuroimaging Initiative](https://adni.loni.usc.edu/)

---

## License
This project is for educational purposes only.

---

## Contact
For questions or issues, please open an issue in the repository.

---

**Key Takeaway**: Distributed training is powerful for scaling deep learning, but requires proper hardware (multiple GPUs) to achieve speedup. This lab demonstrates the concepts using CPU simulation, which helps understand the architecture without expensive GPU infrastructure.
