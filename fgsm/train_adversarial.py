import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from models.cnn import CNN
from utils.training import train_epoch_adversarial
from config import CONFIG
import os

def main():
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Setup data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**CONFIG['transform_stats'])
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=CONFIG['batch_size'],
                                            shuffle=True, 
                                            num_workers=CONFIG['num_workers'])
    
    # Initialize model and training
    model = CNN().to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training loop
    for epoch in range(CONFIG['train_epochs']):
        train_loss = train_epoch_adversarial(
            model, trainloader, criterion, optimizer,
            CONFIG['device'], CONFIG['epsilon'], epoch
        )
        print(f'Epoch {epoch}: Loss = {train_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), 
              f"checkpoints/cifar10_CNN_adv_{CONFIG['train_epochs']}.pth")

if __name__ == "__main__":
    main()
