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

1. **Data preparation**: CIFAR10 is used as in-distribution dataset, CIFAR100 as out-of-distribution one. In particular, only the classes ["Maple_tree", "Aquarium_fish", "Willow_tree", "Flatfish", "Rose", "Lawn_mower", "Porcupine", "Caterpillar", "Seaweed", "Shrew"] have been considered.

2. **Models**:
- CNN: a simple convolutional neural network
- ResNet18 pretrained

3. **Training**: 


In order to distinguish between IN data and OOD data I compute a OOD score using the Maximum Softmax Probability (MSP). Then, I evaluate the performance of OOD detection with histograms, ROC curves and Precision-Recall curves.
<p align="center">
  <img src="https://github.com/user-attachments/assets/6053b905-342c-4824-a571-7be4f3363ee4" width="400"/>
  <img src="https://github.com/user-attachments/assets/da1e1975-c11a-4aba-b83b-5e7c01535f13" width="400"/>
</p> 

<p align="center">
  <img src="https://github.com/user-attachments/assets/9672fc36-0991-40aa-aee5-b7f59d7c20cd" width="400"/>
  <img src="https://github.com/user-attachments/assets/82d30cf3-f270-4382-a4ee-a24b0eae0222" width="400"/>
</p> 

<p align="center">
  <img src="https://github.com/user-attachments/assets/4b35f29e-6f4b-42e3-ab12-c0833d378e6f" width="400"/>
  <img src="https://github.com/user-attachments/assets/c466cc34-9f55-44ab-8421-a2c86cfb2413" width="400"/>
</p> 

<p align="center">
  <img src="https://github.com/user-attachments/assets/5b6cfd12-1dc0-47a0-9539-1ae6130a0b69" width="400"/>
  <img src="https://github.com/user-attachments/assets/47a57fb1-d33f-458d-bff4-4a004ca53685" width="400"/>
</p> 


### cnn_ep{10}
| AUROC | 0.635 |
| AUPR | 0.945 |

### resnet_ep{10}
| AUROC | 0.73 | 
| AUPR | 0.96 | 


## Usage

```bash
python train.py

