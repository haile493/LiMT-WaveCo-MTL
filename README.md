# WaveCo-MTL: Wavelet-Attentive Collaborative Multi-Task Learning for Joint Liver/Tumor Segmentation and Multi-Label Lesion Classification

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

Official PyTorch implementation of **WaveCo-MTL**, a unified multi-task deep learning framework designed for simultaneous liver parenchyma segmentation, focal liver lesion contouring, and 5-class multi-label pathological classification on abdominal CT scans (LiMT Benchmark).

---

## 📌 Key Architectural Highlights

<p align="center">
  <img src="docs/figures/The proposed WaveCo-MTL network.png" width="90%" alt="WaveCo-MTL Architecture">
</p>

- **Dual-Stream Mutual Guidance (Co-Seg)**: Collaborative prompt interaction between liver anatomical context and tumor focal regions.
- **3D Haar Discrete Wavelet Transform (HaarDWT3D)**: High-frequency directional subband decomposition for crisp boundary delineation.
- **Second-Order Bilinear Pooling (SOPSeg)**: Quadratic covariance feature modeling capturing non-linear lesion micro-textures.
- **Masked Feature Pooling (MFP)**: Anatomically guided attention mapping from segmentation decoders directly into the multi-label diagnostic classifier.
- **5-Class Multi-Label Diagnostic Categorization**:
  1. *Class 0*: Liver / Normal Parenchyma
  2. *Class 1*: Hepatic Cyst
  3. *Class 2*: Cavernous Hemangioma
  4. *Class 3*: Hepatic Metastasis
  5. *Class 4*: Hepatocellular Carcinoma (HCC)

---

## 📊 Benchmark Performance on LiMT Test Set

| Model Architecture | Task Paradigm | Liver Dice (%) | Tumor Dice (%) | Cls Accuracy (%) | Cls Precision (%) | Cls Recall (%) | Macro F1 (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet (LiMT Benchmark) | Cls-Only | — | — | 73.66% | 59.23% | 53.27% | 51.59% |
| nnU-Net (LiMT Benchmark) | Seg-Only | 97.03% | 58.80% | — | — | — | — |
| Multi-Task Baseline (Chen et al. [18]) | Multi-Task | 97.03% | 60.04% | 75.83% | 60.91% | 53.03% | 53.40% |
| **WaveCo-MTL (Ours, Calibrated)** | **Multi-Task** | **94.00%** | **66.60%** | **77.33%** | **53.02%** | **64.57%** | **53.37%** |
| **WaveCo Multi-Task Ensemble** | **Multi-Task** | **94.24%** | **66.60%** | **78.67%** | **48.67%** | **52.00%** | **55.97%** |

---

## 📁 Repository Structure

```text
LiMT-WaveCo-MTL/
├── checkpoints/             # Pretrained model weights (best_waveco_mtl.pth)
├── data/
│   ├── dataset.py           # PyTorch multi-task dataset with 3D augmentation & CLAHE
│   ├── limt_splits.csv      # Open stratified 3-way and 5-fold cross-validation partitions
│   └── cls_labels.csv       # Multi-label diagnostic classification labels
├── docs/
│   ├── figures/             # Architectural & benchmark visualization figures
│   └── *.csv                # Benchmark evaluation result tables
├── evaluation/              # Quantitative evaluation scripts
├── models/
│   ├── waveco_mtl_net.py    # Proposed WaveCo-MTL deep architecture
│   ├── coseg_liver.py       # Co-Seg mutual guidance model
│   ├── sopseg_liver.py      # Second-order pooling model
│   ├── waveattn_sopseg_liver.py # Wavelet-attentive model
│   ├── multitask_net.py     # Multi-task baseline architectures
│   └── multitask_msff_net.py# Multi-scale feature fusion network
├── training/
│   └── losses.py            # Combined multi-task loss functions (Dice + BCE)
├── utils/
│   ├── metrics.py           # Segmentation (Dice, IoU, HD95, ASD) & classification metrics
│   └── pam_limt.py          # Post-processing and anatomical analysis
├── train.py                 # Master training entrypoint
├── evaluate.py              # Master evaluation and benchmarking entrypoint
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/hai49/LiMT-WaveCo-MTL.git
cd LiMT-WaveCo-MTL
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the LiMT benchmark dataset from the [official repository](https://github.com/wenzhou-zheng/LiMT) and place the CT scans and segmentation masks into `data/LiMT Dataset/`:

```text
data/LiMT Dataset/
├── ct/                      # NIfTI CT volumes (e.g., limt_001.nii.gz)
├── seg/                     # Ground truth masks (0: Background, 1: Liver, 2: Tumor)
├── cls_labels.csv           # Multi-label lesion diagnoses
└── limt_splits.csv          # Stratified splits
```

### 3. Training

Train the proposed **WaveCo-MTL** network:

```bash
python train.py --model waveco_mtl --epochs 50 --batch_size 4 --lr 3e-4
```

To train single-task or multi-task baselines:

```bash
# Segmentation only
python train.py --model seg_only --epochs 50 --batch_size 4

# Classification only
python train.py --model cls_only --epochs 50 --batch_size 4

# Standard Multi-Task Baseline
python train.py --model multitask_net --epochs 50 --batch_size 4
```

### 4. Evaluation & Benchmarking

Evaluate pre-trained weights on the held-out LiMT test split:

```bash
python evaluate.py --model waveco_mtl --ckpt checkpoints/best_waveco_mtl.pth --split_val test
```

---

## 📄 Citation & Attribution

If you use this codebase or our pre-trained models in your research, please cite:

```bibtex
@article{WaveCoMTL2026,
  title={Multi-Task Learning for Simultaneous Liver and Tumor Segmentation and Multi-Label Lesion Classification: Evaluation on the LiMT Dataset},
  author={Le, et al.},
  journal={submitting},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
