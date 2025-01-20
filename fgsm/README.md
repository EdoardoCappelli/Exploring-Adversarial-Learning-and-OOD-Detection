# FGSM: Fast Gradient Sign Method

This repository implements and evaluates the **Fast Gradient Sign Method (FGSM)**, a common adversarial attack used to test the robustness of machine learning models. Below are the details and instructions for using this repository.

## Directory Structure

```plaintext
fgsm
├── models
│   └── cnn.py      # Convolutional Neural Network definition
├── adv_training.py # Adversarial training script
├── eval.py         # Model evaluation script
├── eval_robust_cnn.py # Evaluate robustness of trained CNN
├── fgsm.py         # FGSM attack implementation
├── results/
├── checkpoints/    # Trained models weights
```

## Key Components
### Model, Optimizer, Dataset Setup
- CrossEntropyLoss as the loss function
- Adam optimizer with a learning rate of 0.0001
- CIFAR10 dataset

### `fgsm.py`
This script implements the FGSM attack, a white-box adversarial attack designed to perturb input images in the direction of the gradient of the loss function, making the model misclassify the inputs. A pre-trained CNN model on CIFAR10 is used for predictions. The main steps are described below:

- Compute the loss using CrossEntropyLoss.
- Calculate the gradient of the loss with respect to the input image.
- The image is perturbed by the gradient's sign, scaled by the specified epsilon.
- The loop continues for a specified number of iterations (max_iter), modifying the image until it misclassifies or exceeds the iterations.


### `adv_training.py`
The CNN is trained on both clean and adversarial examples to increase robustness against adversarial attacks. 

- For each batch, adversarial examples (x_adv) are generated using the Fast Gradient Sign Method (FGSM).
- Compute the loss, perform backpropagation, and adjust the image slightly in the direction of the gradient's sign.
- The model is trained on the adversarial image (x_adv), and the loss is computed.
- The optimizer updates the model's weights based on this adversarial loss.
- The model is also trained on the original images to maintain performance on clean data.
- Another round of optimization is performed using the original images.
  
## Running the FGSM Attack

To execute the FGSM attack on CIFAR-10 using a pre-trained CNN model:

```bash
python fgsm.py --pretrained ./models/cnn.pth --epsilon 0.01 --verbose
```

### Arguments
- `--pretrained`: Path to the pre-trained model weights.
- `--epsilon`: Magnitude of the perturbation.
- `--verbose`: Display visualizations of adversarial examples.

## Usage
### Training

```bash
python adv_training.py --epochs 20 --batch_size 128
```

### Evaluation

```bash
python eval.py --model ./models/cnn.pth
```
### Testing the robustness of the model against adversarial attacks

```bash
python eval_robust_cnn.py --model ./models/cnn.pth --epsilon 0.01
```

## Results

#### CIFAR-10 CNN 
Adversarial attacks performed on CIFAR10 using a simple CNN trained for 30 epochs.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b3298cc7-4fd4-4b8c-af15-a133554f9a2e" width="600"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/7a22abed-9646-4890-9f2c-c3880331822d" width="600"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/b42a890a-b25a-48f8-9bde-4e5500e22a73" width="600"/>
</p>

#### Robust CNN vs Standard CNN

The performance of the attacks and the adversarial training is assessed using specific metrics. The outcomes are presented through the Attack Success Rate (ASR), illustrating how the effectiveness of the attack varies with respect to the epsilon parameter and the maximum number of iterations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2cb23615-ec7b-47c3-95c1-a94ae956a80e" width="800"/>
</p>

---

