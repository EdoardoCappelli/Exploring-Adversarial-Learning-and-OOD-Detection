# CIFAR-10 CNN Classifier

This repository implements a simple Convolutional Neural Network (CNN) to classify images from the **CIFAR-10** dataset. It includes the model definition, training pipeline, and tools to analyze results on in-distribution (ID) and out-of-distribution (OOD) data.

## Project Structure

- `model.py`: Contains the implementation of the CNN architecture.
- `train.py`: Main script for:
  - Loading datasets (**CIFAR-10** and a fake dataset for OOD analysis).
  - Training the CNN model.
  - Saving the trained model.
  - Analyzing logits and comparing in-distribution vs. out-of-distribution results.

## Usage
Run the following command to train the model and analyze results on ID and OOD data:

```python
python train.py --train --epochs 50
```
The trained model is saved in the `checkpoints` directory as `cifar10_CNN.pth`.
The script also provides an histogram comparing logits from ID (CIFAR-10 test set) and OOD (Fake dataset).
