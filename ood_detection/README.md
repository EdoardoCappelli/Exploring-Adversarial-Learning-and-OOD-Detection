# Out-of-Distribution Detection

This project implements a simple Out-of-Distribution (OOD) detection system using a CNN model trained on CIFAR-10. The system can detect whether input samples are from the training distribution (in-distribution) or from a different distribution (out-of-distribution).

## Features

- CNN model implementation for CIFAR-10 classification
- Training and evaluation pipeline
- OOD detection using maximum softmax probability
- Support for different OOD datasets (random noise or CIFAR-100)
- Visualization tools for training curves and OOD detection performance

## Results
### Max Softmax Distribution
![alt text](image.png)

### OOD Detection Evaluation
| AUC ROC| 0.28 | 
| FPR at 95% TPR: 1.0 | 
| AUC PRC: 0.87 | 
| Precision: 0.91 | 
| Recall: 1.0 | 
## Usage

```bash
python train.py

