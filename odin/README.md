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
python grid_search.py --batch_size 256 --ood_set cifar100 --pretrained path_to_pretrained_model.pth
```

### 3. **Explanation of the Parameters**

- `batch_size`: The number of samples in each batch during evaluation.
- `epsilon`: The magnitude of the perturbation applied during input preprocessing.
- `temp`: The temperature parameter used for scaling the logits in the softmax function.
- `ood_set`: Specifies the OOD dataset to evaluate. Options are "fakedata" or "cifar100".
- `pretrained`: The path to the pretrained model.

## Example Output

Running the `test_odin.py` script will output the following metrics:
- **AUC ROC (ODIN)**: The Area Under the ROC Curve for OOD detection.
- **FPR at 95% TPR (ODIN)**: The False Positive Rate when the True Positive Rate is 95%.
- **AUC PRC**: The Area Under the Precision-Recall Curve.
- **Precision**: The precision at the highest recall.
- **Recall**: The recall at the highest precision.

Additionally, plots for the ROC and PR curves will be saved in the `./results` directory.

## Results


## References

- **ODIN**: [Out-of-Distribution Detection with ODIN](https://arxiv.org/abs/1706.02690) by Liang et al.
