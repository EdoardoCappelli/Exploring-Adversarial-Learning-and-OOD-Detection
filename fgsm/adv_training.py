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
from sklearn import metrics
import random
import gc
import os
from utils import NormalizeInverse, evaluate_model
from models.cnn import CNN 


def train_epoch_adv(model, dataloader, loss_fn, optimizer, epoch=0, device='cpu', epsilon=0.01):
  model.train()
  train_loss = []
  bar = tqdm(dataloader, total=len(dataloader))

  for i, (x, y) in enumerate(bar):
    x, y = x.to(device), y.to(device)

    #generate adversarial examples
    x_adv = x.clone().detach().requires_grad_(True)
    for _ in range(1):
      output = model(x_adv)
      loss = loss_fn(output, y)
      model.zero_grad()
      loss.backward()
      with torch.no_grad():
        x_adv += epsilon * torch.sign(x_adv.grad)
      x_adv.requires_grad_(True)

    #train on adversarial examples
    optimizer.zero_grad()
    output_adv = model(x_adv.detach())
    loss_adv = loss_fn(output_adv, y)
    loss_adv.backward()
    optimizer.step()

    #train on original data
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())
    bar.set_description(f"Epoch {epoch} loss: {np.mean(train_loss):.5f}")
  return np.mean(train_loss)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR10 with optional OOD dataset")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--epsilon', type=float, default=0.01, help='perturbation')
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    #train the model with adversarial training
    num_workers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    loss =  nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = args.epochs
    epsilon = args.epsilon
    batch_size = args.batch_size
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for e in range(epochs):
        train_loss = train_epoch_adv(model, trainloader, loss, optimizer, epoch=e, device=device, epsilon=epsilon)
        val_loss, val_acc, val_report = evaluate_model(model, testloader, loss, device=device)
        print(f'Epoch {e} - Validation Loss: {val_loss} - Validation Accuracy: {val_acc}')
        
    # Save model
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f"./checkpoints/robust_cnn_{args.epochs}.pth")
