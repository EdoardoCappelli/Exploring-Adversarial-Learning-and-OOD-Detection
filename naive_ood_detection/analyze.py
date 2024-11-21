import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10, FakeData
from torch.utils.data import DataLoader
from train import CNN

def main():
    # Ensure results directory exists.
    os.makedirs("results", exist_ok=True)

    # Device setup.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset transformations.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 32
    ds_test = CIFAR10(root='data', train=False, download=True, transform=transform)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

    ds_fake = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
    dl_fake = DataLoader(ds_fake, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load the trained model.
    model = CNN().to(device)
    model.load_state_dict(torch.load('checkpoints/cifar10_CNN.pth', weights_only=False))
    model.eval()

    # Collect logits from the model.
    def collect_logits(model, dataloader):
        logits = []
        with torch.no_grad():
            for Xs, _ in dataloader:
                logits.append(model(Xs.to(device)).cpu().numpy())
        return np.vstack(logits)

    # Collect logits for CIFAR-10 and FakeData.
    logits_ID = collect_logits(model, dl_test)
    logits_OOD = collect_logits(model, dl_fake)

    # Plot histograms of max logits.
    plt.figure(figsize=(10, 6))
    plt.hist(logits_ID.max(axis=1), bins=50, density=True, alpha=0.5, label="ID (CIFAR-10)")
    plt.hist(logits_OOD.max(axis=1), bins=50, density=True, alpha=0.5, label="OOD (FakeData)")
    plt.title("Logit Distributions for ID and OOD Samples")
    plt.xlabel("Maximum Logit")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("results/logit_distribution.png")
    plt.show()

    # Plot softmax probabilities.
    softmax_ID = np.exp(logits_ID) / np.sum(np.exp(logits_ID), axis=1, keepdims=True)
    softmax_OOD = np.exp(logits_OOD) / np.sum(np.exp(logits_OOD), axis=1, keepdims=True)

    plt.figure(figsize=(10, 6))
    plt.hist(softmax_ID.max(axis=1), bins=50, density=True, alpha=0.5, label="ID (CIFAR-10)")
    plt.hist(softmax_OOD.max(axis=1), bins=50, density=True, alpha=0.5, label="OOD (FakeData)")
    plt.title("Softmax Probability Distributions for ID and OOD Samples")
    plt.xlabel("Maximum Softmax Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("results/softmax_distribution.png")
    plt.show()

if __name__ == '__main__':
    main()
