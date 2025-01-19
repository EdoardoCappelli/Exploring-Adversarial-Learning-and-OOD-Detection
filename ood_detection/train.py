import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, trainloader, epochs, device, lr=0.001, momentum=0.9):
    """
    Train the model using CUDA if available.
    
    Args:
        model: The neural network model
        trainloader: DataLoader for training data
        epochs: Number of training epochs
        device: Device to train on (cuda or cpu)
        lr: Learning rate
        momentum: Momentum for SGD optimizer
    """
    print(f"Training on {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Enable automatic mixed precision for faster training on CUDA
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Add progress bar for each epoch
        pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}')
        
        for inputs, labels in pbar:
            # Move data to appropriate device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            if device.type == 'cuda':
                # Use automatic mixed precision for faster training
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Scale loss and call backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training on CPU
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.3f}")
    
    print("Finished Training")
    return model

def evaluate_model(model, testloader, device):
    """
    Evaluate the model using CUDA if available.
    
    Args:
        model: The neural network model
        testloader: DataLoader for test data
        device: Device to evaluate on (cuda or cpu)
    """
    print(f"\nEvaluating on {device}")
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.2f}%")
    return accuracy

