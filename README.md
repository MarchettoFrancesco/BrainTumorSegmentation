# 🧠 Brain MRI Tumor Segmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Deep learning-based semantic segmentation of brain tumors in MRI scans using U-Net architectures with state-of-the-art encoder backbones.


<img width="761" height="256" alt="Screenshot 2025-11-25 alle 15 12 12" src="https://github.com/user-attachments/assets/3d51314d-5264-48c5-becf-62ff78e9da00" />


## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Results](#results)


---

## 🎯 Overview

This project implements **binary semantic segmentation** of brain tumors in MRI images using U-Net architectures with various pretrained encoder backbones. The models achieve high accuracy in identifying and delineating tumor regions to assist in medical diagnosis and treatment planning.

### Key Highlights

- ✅ **Multiple U-Net Architectures**: ResNet34, EfficientNet-B2
- ✅ **Comprehensive Training Pipeline**: Automated hyperparameter grid search
- ✅ **Advanced Loss Functions**: BCE + Dice Loss
- ✅ **Model Interpretability**: GradCAM, GradCAM + ABS visualizations
- ✅ **Experiment Tracking**: Weights & Biases (wandb) integration
- ✅ **Medical Image Augmentation**: Albumentations for robust training

---

## ✨ Features

### Training
- **Automated hyperparameter search** over learning rates, batch sizes, and optimizers
  <img width="945" height="442" alt="Screenshot 2025-11-25 alle 15 14 58" src="https://github.com/user-attachments/assets/0b2a94ad-2a24-4049-bfeb-05aa2d7cea34" />

  
- **Multiple optimizer support**: Adam, RMSPROP, SGD and SDG with momentum
- **Learning rate scheduling**: ReduceLROnPlateau + Warmup for stable training
- **Early stopping** mechanism to prevent overfitting
- **Checkpoint management** with automatic best model saving
- **WandB logging** for comprehensive experiment tracking

### Evaluation Metrics
- **Dice Coefficient** (F1-score for segmentation)
- **Intersection over Union (IoU)** - overall and class-wise
- **Pixel-wise Accuracy**
  

### Visualization & Interpretability
- **GradCAM heatmaps** for understanding model focus
- **Prediction vs Ground Truth** comparisons
- **Training curves** and performance plots
- **Absolute gradient GradCAM** for better segmentation interpretation

---

## 📊 Dataset

### Kaggle Brain MRI Segmentation Dataset

**Source**: [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

- **Total Images**: ~3,929 MRI slices from 110 patients
- **Resolution**: Resized to 256×256 pixels
- **Format**: RGB TIFF images
- **Modality**: T1-weighted MRI with contrast enhancement
- **Classes**: Binary segmentation (tumor / non-tumor)
- **Tumor Type**: Low Grade Glioma (LGG)

### Data Preprocessing

1. **Resizing**: All images standardized to 256×256 or 128×128
2. **Normalization**: 
   - Pixel values scaled to [0, 1]
   - Per-image normalization option available
3. **Mask Binarization**: Threshold > 0.5 → 1 (tumor), else 0
4. **Data Split**: 80% train / 10% validation / 10% test
5. **Augmentation**: 
   - Rotation (±10°)
   - Horizontal flip
   - Brightness/contrast adjustment
   - Shift/scale transformations

---

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (for GPU training)

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/MarchettoFrancesco/BrainTumorSegmentation.git
cd BrainTumorSegmentation

# Create conda environment
conda create -n mri_seg python=3.10 -y
conda activate mri_seg

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt

# For CUDA custom operations (optional)
conda install -c conda-forge cudatoolkit-dev -y

```

## 📥 Model Weights

**Note:** Trained model weights (`.pth` files) are not included in this repository due to GitHub's file size limits.

## 📈 Training Results

### Final Model Performance

| Model | Epoch | Val Dice ↑ | Val IoU ↑ | Val Accuracy | Train Dice | Train IoU (BG) | Training Time |
|-------|-------|------------|-----------|--------------|------------|----------------|---------------|
| **EfficentNet-B2** | 35 | **0.8885** | **0.8044** | 0.9980 | 0.7037 | 0.7179 | 14m 9s |
| **ResNet34** | 35 | 0.8838 | 0.7967 | 0.9979 | 0.6989 | 0.7190 | 12m 54s |

<img width="947" height="223" alt="Screenshot 2025-11-25 alle 15 13 22" src="https://github.com/user-attachments/assets/e237f7cc-c51f-4730-8ae4-76f172fc3b34" />

### Performance Metrics Explained

- **Dice Coefficient**: Measures overlap between prediction and ground truth (higher is better)
  - **0.8885** = 88.85% overlap - Excellent performance
- **IoU (Intersection over Union)**: Jaccard index for segmentation quality
  - **0.8044** = 80.44% - Very good segmentation
