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

## 3. **ODIN - Out-of-Distribution Detection Using Deep Learning**

This project implements ODIN, a method for Out-of-Distribution (OOD) detection. The approach enhances traditional OOD detection by leveraging temperature scaling and input gradient-based perturbation to improve the model's decision boundaries and make it more sensitive to OOD samples.

### Features
- ODIN method for OOD detection based on a pre-trained CNN or ResNet model.
- Temperature scaling and gradient-based input perturbation to sharpen the decision boundaries of the model.
- Compatibility with CIFAR-10 for in-distribution data and CIFAR-100 as the OOD dataset.
- Performance evaluation using Maximum Softmax Probability (MSP) and ODIN's temperature-scaled softmax score.
- Integration with existing models (e.g., CNN, ResNet) for OOD detection.

### Key Parameters
- temperature: Scaling factor applied to model logits to adjust model confidence.
- epsilon: Magnitude of the perturbation applied to inputs during preprocessing.
- model: The neural network (e.g., CNN, ResNet) used for classification and OOD detection.

### Usage
To train the model and evaluate OOD detection using the ODIN method:

#### Running ODIN on a Pretrained Model:
```bash
python test_odin.py --batch_size 256 --epsilon 0.01 --temp 0.1 --ood_set cifar100 --pretrained path_to_pretrained_model.pth --verbose
```

#### Grid Search for Hyperparameter Optimization
```bash
python grid_search.py --model-type cnn --batch_size 256 --ood-set cifar100 --model-path checkpoints/cnn_model_ep50 --verbose
```
