import torch
import torchvision
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

        
def evaluate_model(model, dataloader, criterion, device='cpu'):
    model.eval()
    losses = []
    predictions = []
    gts = []
    with torch.no_grad():
        for (inputs, labels) in tqdm(dataloader, desc='Evaluating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            gts.append(labels.cpu().numpy())
            predictions.append(preds.detach().cpu().numpy())
            losses.append(loss.item())
    avg_loss = np.mean(losses)
    accuracy = accuracy_score(np.hstack(gts), np.hstack(predictions))
    report = classification_report(np.hstack(gts), np.hstack(predictions), zero_division=0, digits=3)
    return avg_loss, accuracy, report