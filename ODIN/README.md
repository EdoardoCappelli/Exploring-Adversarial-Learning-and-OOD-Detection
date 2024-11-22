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

## **Implementation Steps**
1. **Load a pre-trained model.**
   ODIN works with existing models trained on in-distribution datasets (e.g., CIFAR-10, CIFAR-100).
   
2. **Adjust the softmax function:**
   Use a modified softmax with a high temperature value.

3. **Add input preprocessing:**
   Introduce small perturbations to the input images using the gradient of the log-softmax score.

4. **Compare softmax scores:**
   Use the modified softmax scores to distinguish in-distribution from OOD images.


## Evaluation
ODIN has been evaluated on various benchmark datasets, showing significant improvements in OOD detection:

### In-distribution datasets
- CIFAR-10
- CIFAR-100

### OOD datasets
- Gaussian Noise
- CIFAR100 subset

### Metrics
- **FPR at 95% TPR:** Measures the false positive rate when true positive rate is 95%.
- **AUROC:** Area Under the Receiver Operating Characteristic curve.

### Results
- ODIN outperforms baseline methods across all tested configurations.
- Parameters tuned on one OOD validation set transfer effectively to others.

## **How to Run ODIN**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/odin-ood-detector.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the ODIN detection script:
   ```bash
   python odin_detector.py --model <model_path> --dataset <dataset_path> --temperature <value> --epsilon <value>
   ```
   Example:
   ```bash
   python odin_detector.py --model resnet.pth --dataset cifar10 --temperature 1000 --epsilon 0.002
   ```


## References
**"Enhancing The Reliability of Out-of-Distribution Image Detection in Neural Networks"**  
Authors: Shiyu Liang, Yixuan Li, and R. Srikant.  
Link: [Paper on arXiv](https://arxiv.org/abs/1706.02690)
