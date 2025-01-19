# Out-of-Distribution Detection

This project implements a simple Out-of-Distribution (OOD) detection system using a CNN model trained on CIFAR-10. The system can detect whether input samples are from the training distribution (in-distribution) or from a different distribution (out-of-distribution).

## Features

- CNN model implementation for CIFAR-10 classification
- Training and evaluation pipeline
- OOD detection using maximum softmax probability
- Support for different OOD datasets (random noise or CIFAR-100)
- Visualization tools for training curves and OOD detection performance

## OOD Detection
The goal is to build a OOD Detector ables to produces a score representing how OOD a test sample is.

1. **Data preparation**: CIFAR10 is used as in-distribution dataset, CIFAR100 as out-of-distribution one. 

2. **Models**:
- CNN: a simple convolutional neural network
- ResNet18 pretrained

3. **Training**:
- CrossEntropyLoss
- Adam optimizer (lr=0.0001)
- GradScaler to enabel automatic mixed precision for faster training on CUDA
- epochs [10, 50]
- trained models [cnn, resnet]

## How to distinguish between IN and OOD data?
In order to distinguish between IN data and OOD data an OOD we can implement the following strategies:
- **Maximum Softmax Probability (MSP) score**
  
| CNN | RESNET |
|-----|--------|
| <p align="center"><img src="https://github.com/user-attachments/assets/4d4c365b-6c73-47b5-9e21-9551c370ba70" width="400"/></p> | <p align="center"><img src="https://github.com/user-attachments/assets/18ca3d6a-3097-4919-b7a9-68b701bd7ec7" width="400"/></p> |
| <p align="center"><img src="https://github.com/user-attachments/assets/cc5abbd0-5cf0-4340-adb6-fe969a129a46" width="400"/></p> | <p align="center"><img src="https://github.com/user-attachments/assets/35c5e019-8aa0-4f31-8950-5c5e72cd3f46" width="400"/></p> |

For ID data, the scores tend to be higher, as the model recognizes them and is more confident in its predictions.
For OOD data, the scores tend to be lower, as the model does not recognize them and is therefore less confident.

- **ROC and Precision-Recall curves**

| CNN |
|-----|
| <p align="center"><img src="https://github.com/user-attachments/assets/5a73a889-b838-4518-b27e-672d3a54e9d9" width="1000"/></p> |

| RESNET |
|-----|
| <p align="center"><img src="https://github.com/user-attachments/assets/7636917a-92ac-4a0c-8d05-baed3a642cdc" width="1000"/></p> |

- **Metrics summary**
  
| Metric            | CNN   | RESNET |
|-------------------|-------|--------| 
| Accuracy on test  | 0.71  | 0.82  |
| AUC               | 0.67  | 0.75   |
| FPR at 95% TPR    | 0.91  | 0.84   |
| AUPR              | 0.97  | 0.98   |


## Usage

```bash
python train.py

