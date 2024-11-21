import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from model import CNN 
from torchvision.datasets import FakeData, CIFAR10
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def get_dataloader():
    # We will use CIFAR-10 as our in-distribution dataset.
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the datasets and setup the DataLoaders.
    batch_size = 32
    ds_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)

    ds_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

    # In case we want to pretty-print classifications.
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Fake dataset.
    ds_fake = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
    dl_fake = torch.utils.data.DataLoader(ds_fake, batch_size=batch_size, shuffle=False, num_workers=2)

    # Plot a fake image.
    plt.imshow(FakeData(size=1, image_size=(3, 32, 32))[0][0])

    return ds_fake, dl_fake, ds_train, dl_train, ds_test, dl_test

def collect_logits(model, dl):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = []
    with torch.no_grad():
        for (Xs, _) in dl:
            logits.append(model(Xs.to(device)).cpu().numpy())
    return np.vstack(logits)


def analyze_ID_OOD(model, dl_test, dl_fake):
    # Collect logits on CIFAR-10 test set (ID) and noise (very OOD).
    logits_ID = collect_logits(model, dl_test)
    logits_OOD = collect_logits(model, dl_fake)

    # Plot the *distribution* of max logit outputs.
    _ = plt.hist(logits_ID.max(1), 50, density=True, alpha=0.5, label='ID')
    _ = plt.hist(logits_OOD.max(1), 50, density=True, alpha=0.5, label='OOD')
    plt.legend()
    
def train_model(epochs, dl_train):
    # Instantiate the model and move it to the device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=> Start training using {device}")
    model = CNN().to(device)
    
    # Define loss criterion and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Main training loop.
    for epoch in range(epochs):
        running_loss = 0.0
        # Iterate over all batches.
        for i, (Xs, ys) in enumerate(dl_train, 0):
            Xs = Xs.to(device)
            ys = ys.to(device)
            
            # Make a gradient step.
            optimizer.zero_grad()
            outputs = model(Xs)
            loss = criterion(outputs, ys)
            loss.backward()
            optimizer.step()
            
            # Track epoch loss.
            running_loss += loss.item()
        
        # Print average epoch loss.
        print(f'{epoch + 1} loss: {running_loss / len(dl_train):.3f}')
    
    print('Finished Training')
    # Save the model.
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), './checkpoints/cifar10_CNN.pth')

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set up argument parsing.
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 dataset.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train (default: 50)")
    
    args = parser.parse_args()
    ds_fake, dl_fake, ds_train, dl_train, ds_test, dl_test = get_dataloader()
    train_model(args.epochs, dl_train)
    model = CNN().to(device)
    model.load_state_dict(torch.load('./checkpoints/cifar10_CNN.pth'))  
