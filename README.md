# Exploring-Adversarial-Learning-and-OOD-Detection

This repository contains three distinct projects related to OOD detection and adversarial learning.

---

## 1. **Out-of-Distribution (OOD) Detection**

This project implements a system to detect Out-of-Distribution (OOD) samples using a Convolutional Neural Network (CNN) trained on CIFAR-10. The goal is to distinguish in-distribution (ID) data from OOD data effectively.

### Features

- CNN and ResNet18 models for CIFAR-10 classification and OOD detection.
- OOD detection using Maximum Softmax Probability (MSP) scores.
- CIFAR-100 as OOD dataset
- Performance metrics including ROC and Precision-Recall curves.

### Usage

To train and evaluate the OOD detection pipeline:

```bash
python main.py --epochs 50 --model-type cnn
```

---

## 2. **Fast Gradient Sign Method (FGSM) Attacks**

This module implements the Fast Gradient Sign Method (FGSM) for generating adversarial examples against neural networks. The implementation includes both single-step and iterative variants of FGSM.

### Features

- Basic FGSM implementation with configurable epsilon values
- Iterative FGSM with customizable maximum iterations
- Support for both robust and standard model evaluation
- Visualization tools for attack success rates

### Key Parameters

- `epsilon`: Perturbation magnitude (default values: 0.001 to 0.1)
- `max_iter`: Maximum number of iterations for iterative FGSM (default values: 1 to 50)
- Attack success rate (ASR) evaluation against both robust and standard models

### Usage

To evaluate FGSM attacks against both robust and standard models:

```bash
python fgsm_evaluation.py --robust-model path/to/robust_model.pth --standard-model path/to/standard_model.pth
```
