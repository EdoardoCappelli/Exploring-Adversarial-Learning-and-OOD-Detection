# Exploring-Adversarial-Learning-and-OOD-Detection

This repository contains three distinct projects releted to OOD detection and adversarial learning.

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

## 2. **Adversarial Attacks**

This project explores the implementation and evaluation of adversarial attack methods on machine learning models. It aims to understand model vulnerabilities and improve their robustness against adversarial inputs.

### Features

- Support for multiple attack algorithms, including FGSM and PGD.
- Implementation of targeted and untargeted attacks.
- Visualizations to demonstrate adversarial perturbations.
- Analysis of model performance under attack.

### Usage

Run the main script to generate adversarial examples and evaluate their impact:

```bash
python adversarial_attack.py --attack-type pgd --epsilon 0.03
```

---

## 3. **ODIN: Out-of-Distribution Detector**

This project implements the ODIN technique for OOD detection. ODIN uses temperature scaling and input perturbation to improve the separation between in-distribution and out-of-distribution samples.

### Features

- Implementation of temperature scaling and input perturbation.
- Support for multiple OOD datasets.
- Detailed performance analysis using metrics like AUC and FPR.
- Visualization of detection performance.

### Usage

To run the ODIN method for OOD detection:

```bash
python odin.py --temperature 1000 --epsilon 0.0014
```

---

## General Notes

Each project contains a `README.md` file with detailed instructions, explanations, and usage examples. To get started, navigate to the respective folder and follow the instructions provided.

Feel free to clone the repository and explore the projects:

```bash
git clone https://github.com/your-repo-name.git
