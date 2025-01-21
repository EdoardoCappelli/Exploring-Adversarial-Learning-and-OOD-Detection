# ODIN: Enhancing the Reliability of Out-of-Distribution Image Detection in Neural Networks

## Introduction

While neural networks are effective at classifying images within their training distribution, they often make high-confidence predictions for out-of-distribution (OOD) images, leading to misclassifications. The ODIN method, introduced in the [paper](https://arxiv.org/abs/1706.02690), addresses this issue by improving OOD detection without requiring the retraining of the neural network. The method enhances the separability between in-distribution (ID) and OOD images using two key components:

1. **Temperature Scaling**: This technique involves scaling the logits produced by the network before applying the softmax function. It helps separate the softmax score distributions for ID and OOD images, making OOD detection more effective.
   
2. **Input Preprocessing**: ODIN adds small, controlled perturbations to the input image to increase the softmax score for ID images. The perturbations affect ID images more strongly than OOD images, which improves the model's ability to distinguish between the two.

This repository implements the ODIN method for OOD detection.

## File Descriptions

- **odin.py**: Contains the implementation of the ODIN method, which includes the model, temperature scaling, gradient computation, and input preprocessing steps. This file defines the `Odin` class that can be instantiated and used to detect OOD images.

- **test_odin.py**: Provides a script for testing the ODIN method on different datasets, including CIFAR-10 and CIFAR-100. It includes functionality for evaluating OOD detection performance using Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves, as well as computing metrics like AUC (Area Under the Curve).

- **grid_search.py**: Implements a grid search to find the best combination of temperature and epsilon values for ODIN. It evaluates the ODIN method on a given test set and an OOD dataset, calculating the AUC score to determine the optimal parameters.

## Usage

### 1. **Running ODIN on a Pretrained Model**

To test ODIN with a pretrained model, use the `test_odin.py` script. You need to specify the batch size, epsilon, temperature, and the dataset to evaluate. Additionally, provide the path to the pretrained model.

Example usage:
```bash
python test_odin.py --batch_size 256 --epsilon 0.01 --temp 0.1 --ood_set cifar100 --pretrained path_to_pretrained_model.pth --verbose
```

This will evaluate the ODIN method on the CIFAR-100 OOD dataset, using the pretrained model located at `path_to_pretrained_model.pth`, with an epsilon of `0.01` and a temperature of `0.1`.

### 2. **Grid Search for Hyperparameter Optimization**

To perform a grid search to find the best temperature and epsilon values, use the `grid_search.py` script. It will search through a predefined set of temperature and epsilon values and report the best combination based on the AUC score.

Example usage:
```bash
python grid_search.py --model-type cnn --batch_size 256 --ood-set cifar100 --model-path checkpoints\cnn_model_ep50 --verbose
```
### Grid Search results
The follwing parameters were choosen:
- temperatures = [1, 50, 100]
- epsilons = [0.01, 0.05, 0.1]

| Temperature | Epsilon | AUC     | | Temperature | Epsilon | AUC     |
|-------------|---------|---------|-|-------------|---------|---------|
| 1           | 0.01    | 0.7004  | | 50          | 0.03    | 0.7414  |
| 1           | 0.02    | 0.7104  | | 50          | 0.04    | 0.7467  |
| 1           | 0.03    | 0.7180  | | 100         | 0.01    | 0.7263  |
| 1           | 0.04    | 0.7236  | | 100         | 0.02    | 0.7341  |
| 10          | 0.01    | 0.7297  | | 100         | 0.03    | 0.7408  |
| 10          | 0.02    | 0.7378  | | 100         | 0.04    | 0.7461  |
| 10          | 0.03    | 0.7445  | | 200         | 0.01    | 0.7260  |
| **10**          | **0.04**    | **0.7498**  ||  200         | 0.02    | 0.7338  |
| 20          | 0.01    | 0.7284  | | 200         | 0.03    | 0.7404  |
| 20          | 0.02    | 0.7363  | | 200         | 0.04    | 0.7457  |
| 20          | 0.03    | 0.7429  | | 500         | 0.01    | 0.7258  |
| 20          | 0.04    | 0.7483  | | 500         | 0.02    | 0.7336  |
| 50          | 0.01    | 0.7269  | | 500         | 0.03    | 0.7402  |
| 50          | 0.02    | 0.7348  | | 500         | 0.04    | 0.7455  |



**Best Results:**
- **Temperature:** 10
- **Epsilon:** 0.04
- **AUC:** 0.7498

## Results

```bash
python test_odin.py --model-type cnn --ood-set cifar100 --epsilon 0.01 --temp 1 --model-path checkpoints\cnn_model_ep50 --verbose 
```

| Metric                  | Value  |
|-------------------------|--------|
| AUC ROC (ODIN)          | 0.75   |
| FPR at 95% TPR (ODIN)   | 0.78   |
| AUC PRC                 | 0.91   |
| Precision               | 0.77   |
| Recall                  | 1.00   |

### FRP-TPR and Precision-Recall curves
<p align="center">
  <img src="https://github.com/user-attachments/assets/79b4802c-4c1e-46b2-94dd-fac51ff0072a" width="600"/>
  <img src="https://github.com/user-attachments/assets/9b9675d9-f037-4f8b-a611-fa4491706fe3" width="600"/>
</p>


## References

- **ODIN**: [Out-of-Distribution Detection with ODIN](https://arxiv.org/abs/1706.02690) by Liang et al.
