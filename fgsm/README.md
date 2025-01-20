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
```

## Key Components

### `fgsm.py`
This script implements the FGSM attack, a white-box adversarial attack designed to perturb input images in the direction of the gradient of the loss function, making the model misclassify the inputs.

#### Main Functions
- **`fgsm_attack`**: Generates adversarial examples by adding perturbations to the input image.
- **`show_attack`**: Visualizes the original image, adversarial image, and the perturbation applied.

#### Script Highlights
- The attack iteratively updates the input image based on the sign of the gradient.
- It uses a pre-trained CNN model for predictions.
- Supports saving and optionally displaying adversarial examples.

### `adv_training.py`
This script facilitates adversarial training, a method to improve model robustness by incorporating adversarial examples during the training process.

### `eval.py`
Provides tools for evaluating the performance of the model on clean and adversarial samples.

## Running the FGSM Attack

To execute the FGSM attack on CIFAR-10 using a pre-trained CNN model:

```bash
python fgsm.py --pretrained ./models/cnn.pth --epsilon 0.01 --verbose
```

### Arguments
- `--pretrained`: Path to the pre-trained model weights.
- `--epsilon`: Magnitude of the perturbation.
- `--verbose`: Display visualizations of adversarial examples.

### Example Output
The script will output the following:
- Iteration count for successful attacks.
- Visualization of the original, adversarial, and difference images.
- Saved adversarial examples in the `results` directory.


## Adversarial Training
To train a model with adversarial examples for improved robustness:

```bash
python adv_training.py --epochs 20 --batch_size 128
```

## Evaluation
Evaluate the performance of the trained model on adversarial samples:

```bash
python eval.py --model ./models/cnn.pth
```

Test the robustness of the model against adversarial attacks:

```bash
python eval_robust_cnn.py --model ./models/cnn.pth --epsilon 0.01
```
### Results

#### CIFAR-10 CNN (cifar10_cnn_30)

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

