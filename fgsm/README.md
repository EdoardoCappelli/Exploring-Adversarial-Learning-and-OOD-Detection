# Adversarial Attacks with FGSM

This project implements Fast Gradient Sign Method (FGSM) adversarial attacks on a CNN model trained on CIFAR-10. It includes both attack implementation and adversarial training as a defense mechanism.

## Features

- FGSM attack implementation
- Adversarial training
- Attack visualization
- Evaluation metrics for attack success rate
- Support for different attack parameters (epsilon, iterations)

## Usage

1. Train a simple CNN with adversarial training:
```bash
python train_adversarial.py
```

3. Visualize attacks:
```bash
python visualize_attack.py
```

2. Evaluate attacks on the model:
```bash
python evaluate_attacks.py
```

The scripts will save results and visualizations in the `results/` directory.


## Attack Parameters

- Epsilon (ε): Controls the magnitude of the perturbation
- Max Iterations: Number of attack iterations
- Both can be modified in config.py

## Results
### Attack Visualization


## Attacks Evaluation

The evaluation produces two main metrics:
1. Attack Success Rate (ASR) vs Epsilon


2. Attack Success Rate (ASR) vs Number of Iterations


## Reference

"Explaining and Harnessing Adversarial Examples" by Goodfellow et al.
https://arxiv.org/abs/1412.6572
