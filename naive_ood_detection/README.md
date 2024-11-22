# Out-Of-Distribution Data Detection

This repository contains code to set up an experiment that analyzes how a simple Convolutional Neural Network (CNN), trained on the CIFAR-10 dataset, behaves when presented with Out-of-Distribution (OOD) samples.

## Project Structure

- `model.py`: Contains the implementation of the CNN architecture.
- `train.py`: Main script for:
  - Loading datasets (**CIFAR-10** and a fake dataset for OOD analysis).
  - Training the CNN model.
  - Saving the trained model.
- `analize.py`: Main script for:
  
  - Analyzing logits and comparing in-distribution vs. out-of-distribution results.

## Usage
### Training the model 
Run the following command to train the model on CIFAR-10:

```python
python train.py --batch_size 64 --epochs 30 --learning_rate 0.0005 --checkpoint_dir "checkpoints"
```
The trained model is saved in the `checkpoint_dir` directory as `cifar10_CNN.pth`.

### Logit Collection and Visualization

The script also provides an histogram comparing logits from ID (CIFAR-10 test set) and OOD (Fake dataset or CIFAR-100).

Run the following command to analyze results on ID and OOD data:

```python
python analyze.py --use_real_ood --checkpoints "path/to/checkpoints.pth"
```

## Is looking at the max logit the *best* we can do using *just* the CNN outputs? Is there maybe a better way to try to gauge model *confidence* from the logits?
The max logit is a simple heuristic to estimate model confidence. However, other methods might provide a more robust measure of uncertainty, for example ODIN (Out-of-Distribution Detector for Neural Networks).

ODIN involves adding a small perturbation to the input and re-evaluating the logits. By analyzing the changes, one can better differentiate OOD from ID samples.


## Does the behavior of the network on OOD data get *better* or *worse* with more (or fewer) training epochs? 
- Fewer Epochs: The model is underfitted and it may generalize poorly and fail to detect OOD samples, as the learned features are insufficient.
- More Epochs: Overfitting to the ID data can lead to overly confident predictions on OOD data, as the model becomes biased toward ID features.

Let's experiment with models trained for different epochs (10, 50, 100) and compare their OOD performance:
After 10 epochs:

 <p align="center">
  <img width="420" src="https://github.com/user-attachments/assets/516f82d8-6e46-4771-a4d9-ba52cf7a4dd2">
  <img width="420" src="https://github.com/user-attachments/assets/b0b73c63-f480-4643-b250-5e20fbddb3b4">
</p>

After 50 epochs:
<p align="center">
  <img width="420" src="https://github.com/user-attachments/assets/ac9ac91e-4caf-467b-9edd-65a3c257c696">
  <img width="420" src="https://github.com/user-attachments/assets/9175492a-4723-4529-9d4d-592a13d59119">
</p>

 After 100 epochs:

<p align="center">
  <img width="420" src="https://github.com/user-attachments/assets/d8114486-759b-4f58-9b77-48910d0c5761">
  <img width="420" src="https://github.com/user-attachments/assets/f01f128d-920a-46a2-9e82-ea35735da249">
</p>

The logits distribution for the FakeData suggest that, as we expected, the model doesn't have high confidence in its prediction.
On the contrary, the logits distribution for the ID data suggest that the model has greater confidence in its predictions. The long right tail suggests that for some classes, the model is very confident.

## Does the problem get worse if we test using *real* images as OOD samples?

Let's find a subset of CIFAR-100 classes that are *distinct* from those in CIFAR-10 and test this theory.

<!---
<p align="center">
  <img width="420" src="https://github.com/user-attachments/assets/b9a4af40-4eb3-46e2-8fc1-6c12fe17848d">
  <img width="420" src="https://github.com/user-attachments/assets/9b3eecec-f26f-4aa0-baea-b646955a912c">
</p>
--> 
 

<p align="center">
  <img width="420" src="https://github.com/user-attachments/assets/4f16dd61-6db3-4029-a5ec-7f71b4f147d2">
  <img width="420" src="https://github.com/user-attachments/assets/db863879-894c-4200-979a-7cfcd060d800">
</p>

<!---
<p align="center">
  <img width="420" src="https://github.com/user-attachments/assets/7a94d533-d556-4cab-bc19-aba6fc97296f">
  <img width="420" src="https://github.com/user-attachments/assets/98d20a4f-527e-4828-abfe-0536b467ea0e">
</p>
--> 

