# Out-Of-Distribution Data Detection

This repository contains code to set up an experiment that analyzes how a simple Convolutional Neural Network (CNN), trained on the CIFAR-10 dataset, behaves when presented with Out-of-Distribution (OOD) samples.

## Project Structure

- `model.py`: Contains the implementation of the CNN architecture.
- `train.py`: Main script for:
  - Loading datasets (**CIFAR-10** and a fake dataset for OOD analysis).
  - Training the CNN model.
  - Saving the trained model.
  - Analyzing logits and comparing in-distribution vs. out-of-distribution results.

## Usage
Run the following command to train the model:

```python
python train.py --train --epochs 50
```
The trained model is saved in the `checkpoints` directory as `cifar10_CNN.pth`.
The script also provides an histogram comparing logits from ID (CIFAR-10 test set) and OOD (Fake dataset).

Run the following command to analyze results on ID and OOD data:

## Is looking at the max logit the *best* we can do using *just* the CNN outputs? Is there maybe a better way to try to gauge model *confidence* from the logits?


## Does the behavior of the network on OOD data get *better* or *worse* with more (or fewer) training epochs? 


## Does the problem get worse if we test using *real* images as OOD samples? Find a subset of CIFAR-100 classes that are *distinct* from those in CIFAR-10 and test this theory.



