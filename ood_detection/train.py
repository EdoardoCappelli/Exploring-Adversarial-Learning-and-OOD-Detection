import torch
import torchvision
from torchvision.datasets import FakeData
from torch.utils.data import Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
import random
import gc
import os
from utils import evaluate_model, train_epoch
from models.cnn import CNN 
import argparse
import torch
import torchvision
from torchvision.datasets import FakeData
from torch.utils.data import Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
import random
import gc
import os
from utils import evaluate_model, train_epoch

  
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR10 with optional OOD dataset")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--ood_set', type=str, choices=['fakedata', 'cifar100'], default='fakedata', help='Out-of-distribution dataset')
    args = parser.parse_args()

    # Data transformation
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # Settings
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2
    id_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # OOD dataset selection
    if args.ood_set == 'fakedata':
        fakeset = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
        fakeloader = torch.utils.data.DataLoader(fakeset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if args.ood_set == 'cifar100':
        ood_classes = ["aquarium", "bycicle", "bottle", "bed", "rocket", "can", "girl", "chair"]
        cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        selected_indices = [i for i, (_, label) in enumerate(cifar100) if cifar100.classes[label] in ood_classes]
        fakeset = Subset(cifar100, selected_indices)
        fakeloader = torch.utils.data.DataLoader(fakeset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Print settings
    print("==== SETTINGS ====")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}")
    print(f"OOD Dataset: {args.ood_set}")
    print("==================")

    # Initialize model, loss, and optimizer
    model = CNN().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    epochs = args.epochs
    for e in range(epochs):
        train_epoch(model, trainloader, loss, optimizer, epoch=e, device=device)
        val_loss, val_acc, val_report = evaluate_model(model, testloader, loss, device=device)
        print(f'Epoch {e} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f}')

    # Save model
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f"./checkpoints/cifar10_cnn_{args.epochs}.pth")
