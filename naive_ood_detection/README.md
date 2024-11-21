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
### Training the model 
Run the following command to train the model on CIFAR-10:

```python
python train.py --batch_size 64 --epochs 30 --learning_rate 0.0005 --checkpoint_dir "checkpoints"
```
The trained model is saved in the `checkpoint_dir` directory as `cifar10_CNN.pth`.

### Logit Collection and Visualization

The script also provides an histogram comparing logits from ID (CIFAR-10 test set) and OOD (Fake dataset).

Run the following command to analyze results on ID and OOD data:

## Is looking at the max logit the *best* we can do using *just* the CNN outputs? Is there maybe a better way to try to gauge model *confidence* from the logits?
The max logit is a simple heuristic to estimate model confidence. However, other methods might provide a more robust measure of uncertainty, for example ODIN (Out-of-Distribution Detector for Neural Networks).

ODIN involves adding a small perturbation to the input and re-evaluating the logits. By analyzing the changes, one can better differentiate OOD from ID samples.


## Does the behavior of the network on OOD data get *better* or *worse* with more (or fewer) training epochs? 
- Fewer Epochs: The model is underfitted and it may generalize poorly and fail to detect OOD samples, as the learned features are insufficient.
- More Epochs: Overfitting to the ID data can lead to overly confident predictions on OOD data, as the model becomes biased toward ID features.

Let's experiment with models trained for different epochs (e.g., 10, 50, 100 epochs) and compare their OOD performance using metrics like AUROC (Area Under the Receiver Operating Characteristic).



## Does the problem get worse if we test using *real* images as OOD samples? Find a subset of CIFAR-100 classes that are *distinct* from those in CIFAR-10 and test this theory.

<p align="center">
  <img width="720" src="https://github.com/user-attachments/assets/8fc12753-d1c8-4d5e-b762-f604eec3ce3f">
</p>

<p align="center">
  <img width="720" src="https://github.com/user-attachments/assets/2ba06036-8bd8-40aa-bd9a-1961c0d27b0e">
</p>
