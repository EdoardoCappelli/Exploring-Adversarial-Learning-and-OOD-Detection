
import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import CNN
from utils.metrics import max_softmax, compute_scores, evaluate_ood_detection
from utils.data_loaders import get_dataloaders
from utils.training import train_epoch, evaluate_model
from utils.plotting import plot_validation_curves, plot_ood_distributions
from configs.config import CONFIG

def main():
    # Create checkpoint directory
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    
    # Get dataloaders
    trainloader, testloader, oodloader = get_dataloaders(ood_type=CONFIG['data_loader'])
    
    # Initialize model, loss, and optimizer
    model = CNN().to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training loop
    for epochs in CONFIG['train_epochs']:
        checkpoint_path = f"{CONFIG['checkpoint_dir']}/cifar10_CNN_{epochs}.pth"
        
        if os.path.exists(checkpoint_path):
            continue
            
        losses_and_accs = []
        for e in range(epochs):
            train_loss = train_epoch(model, trainloader, criterion, optimizer, 
                                   epoch=e, device=CONFIG['device'])
            val_loss, val_acc, val_report = evaluate_model(
                model, testloader, criterion, device=CONFIG['device'])
            losses_and_accs.append((train_loss, val_loss, val_acc))
            print(f'Epoch {e} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
        
        # Save model and plot curves
        torch.save(model.state_dict(), checkpoint_path)
        plot_validation_curves(losses_and_accs, 
                             f"{CONFIG['checkpoint_dir']}/training_curves_{epochs}.png")
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(
        f"{CONFIG['checkpoint_dir']}/cifar10_CNN_10.pth", 
        map_location=CONFIG['device'], weights_only=True))
    
    # Evaluate on test set
    _, test_acc, report = evaluate_model(model, testloader, criterion, 
                                       device=CONFIG['device'])
    print(f'Test Accuracy: {test_acc}')
    print(report)
    
    # OOD Detection
    scores_id = compute_scores(model, testloader, max_softmax, CONFIG['device'])
    scores_ood = compute_scores(model, oodloader, max_softmax, CONFIG['device'])
    
    # Plot distributions
    plot_ood_distributions(scores_id, scores_ood)
    
    # Evaluate OOD detection
    metrics = evaluate_ood_detection(scores_id, scores_ood)
    print("\nOOD Detection Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
