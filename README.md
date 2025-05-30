# CGAN-CraftEnhancer

This project implements a Conditional Generative Adversarial Network (CGAN) using PyTorch to **enhance or correct poor predictions in craft-related images** from the IDD dataset. The model leverages a U-Net-based Generator and a pre-trained WideResNet-based Discriminator to generate improved image outputs when the original predictions are missing or incorrect.

---

## 🎯 Project Goal

To develop a GAN-based enhancement pipeline that:
- Detects and corrects low-quality predictions in craft images
- Improves prediction reliability in autonomous systems or annotated datasets
- Demonstrates how CGANs can learn from label cues and high-quality references

---

## 🛠️ Features

- U-Net-based Generator architecture
- WideResNet-50 (pretrained) used as a Discriminator
- Dynamic image selection from `.json` flag-based annotations
- Matplotlib for training loss visualization
- Auto-saving of generated images after each epoch
- Support for CUDA-enabled training

---

## 🧠 Architecture Overview

### Generator (U-Net):
- Encoder-decoder structure with skip connections
- Learns to transform poor predictions into corrected outputs

### Discriminator (WideResNet):
- Fine-tuned on high-quality images to distinguish real from generated

### Training Objectives:
- **Discriminator** minimizes Binary Cross-Entropy loss
- **Generator** minimizes adversarial loss + reconstruction MSE

---

## 📂 Dataset Assumptions

The project assumes you are using an IDD-like dataset with:
- Raw images in `/IDD_dataset/no_box/`
- JSON files containing flags like `missing_prediction`, `improper_prediction`, and `correct_prediction`

Images are split into:
- `generator_data`: For training the Generator (incorrect or missing predictions)
- `discriminator_data`: For training the Discriminator (correct predictions)

---

## 💾 Setup

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA)
- torchvision
- OpenCV
- matplotlib
- tqdm

### Installation

```bash
pip install torch torchvision opencv-python matplotlib tqdm
