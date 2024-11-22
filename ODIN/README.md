# ODIN: Out-of-DIstribution Detector for Neural Networks

## **Overview**

ODIN is a simple yet effective method for detecting **Out-of-Distribution (OOD)** images in neural networks without requiring any retraining. It enhances the ability of a neural network to distinguish between **in-distribution** (data similar to the training dataset) and **OOD** inputs (data from a different distribution). This is achieved through two main techniques: **Temperature Scaling** and **Input Preprocessing**.

## **How ODIN Works**
ODIN leverages the following components to improve OOD detection:

### **1. Temperature Scaling**
- **Purpose:** Adjusts the **temperature** parameter of the softmax function to improve separation between in-distribution and OOD images.
- **How it works:**
  - The softmax function is modified to include a temperature parameter, $T$.  
    $$\text{Softmax}(z_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
    where $z_i$ represents the unnormalized output logits, and $T > 1$ increases the separation between softmax scores.
  - A higher temperature reduces the influence of large logits, smoothing the probability distribution and highlighting differences between in-distribution and OOD images.

### **2. Input Preprocessing**
- **Purpose:** Introduces small, controlled **perturbations** to the input image to amplify the softmax score differences between in-distribution and OOD images.
- **How it works:**
  - Perturbations are calculated using the gradient of the log-softmax score with respect to the input:
    $$\tilde{x} = x - \epsilon \cdot \text{sign} \left( \nabla_x \log \text{Softmax}(z_{true}) \right)$$
    where $\epsilon$ is a small perturbation magnitude, and $z_{true}$ is the logit for the predicted class.
  - These perturbations have a stronger effect on in-distribution images than on OOD images, further improving separability.

---
 
## In- and Out-Of-Distribution datasets
- ID
  - CIFAR-10
- OOD
  - Gaussian Noise
  - CIFAR100 subset (no CIFAR-10 classes)
 
  
### Metrics 
The metrics used to evaluate how good a neural network is in distinguishing in- and out-of-distribution images are:
- **FPR at 95% TPR:** Measures the false positive rate when true positive rate is 95%.
- **AUROC:** Area Under the Receiver Operating Characteristic curve.

### Results
 
## How to Run ODIN
 
 ```bash
 python odin_detector.py --model <model_path> --dataset <dataset_path> --temperature <value> --epsilon <value>
 ```
 
## References
**"Enhancing The Reliability of Out-of-Distribution Image Detection in Neural Networks"**  
Authors: Shiyu Liang, Yixuan Li, and R. Srikant.  
Link: [Paper on arXiv](https://arxiv.org/abs/1706.02690)
