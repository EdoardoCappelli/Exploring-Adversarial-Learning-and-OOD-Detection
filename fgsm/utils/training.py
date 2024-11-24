import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train_epoch_adversarial(model, dataloader, criterion, optimizer, device, epsilon=0.01, epoch=0):
    """Training loop with adversarial training"""
    model.train()
    train_losses = []
    
    for x, y in tqdm(dataloader, desc=f"Epoch {epoch}"):
        x, y = x.to(device), y.to(device)

        # Generate adversarial examples
        x_adv = x.clone().detach().requires_grad_(True)
        output = model(x_adv)
        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            x_adv += epsilon * torch.sign(x_adv.grad)
        
        # Train on adversarial examples
        optimizer.zero_grad()
        output_adv = model(x_adv.detach())
        loss_adv = criterion(output_adv, y)
        loss_adv.backward()
        optimizer.step()

        # Train on clean examples
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    return np.mean(train_losses)