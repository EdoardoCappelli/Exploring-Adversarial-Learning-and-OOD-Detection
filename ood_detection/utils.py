import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs('results', exist_ok=True)

def max_logit(logit):
    s, _ = logit.max(dim=1) 
    return s

def max_softmax(logit, T=1.0):
    s = F.softmax(logit/T, 1)
    s, _ = s.max(dim=1)  
    return s

def compute_scores(data_loader, score_function, model, device):
    """Compute scores for a dataset."""
    scores = []
    with torch.no_grad():
        for data in data_loader:
            x, y = data
            output = model(x.to(device))
            s = score_function(output)
            scores.append(s.cpu())
        scores_t = torch.cat(scores)
        return scores_t

def plot_distributions(scores_id, scores_ood, title=""):
    """Plot score distributions for ID and OOD data."""
    ensure_results_dir()
    
    plt.figure()
    plt.hist(scores_id.cpu().numpy(), density=True, bins=25, alpha=0.5, 
             label="ID Data (CIFAR-10)", color='blue')
    plt.hist(scores_ood.cpu().numpy(), density=True, bins=25, alpha=0.5, 
             label="OOD Data (CIFAR-100)", color='red')
    plt.title(f"Score Distribution for ID and OOD Data")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Save the plot with the model name
    plt.savefig(f'results/distributions_{title}.png')
    plt.close()

def plot_sorted_scores(scores_test, scores_fake, title=""):
    """Plot sorted scores for ID and OOD data."""
    ensure_results_dir()
    
    # Sort the scores
    sorted_scores_test = sorted(scores_test.cpu().numpy())
    sorted_scores_fake = sorted(scores_fake.cpu().numpy())
    
    # Plot the sorted scores
    plt.figure()
    plt.plot(sorted_scores_test, label="Test Data (CIFAR-10)", color='blue')
    plt.plot(sorted_scores_fake, label="Fake Data (CIFAR-100)", color='red')
    plt.title(f"Sorted Scores for ID and OOD Data")
    plt.xlabel("Sample Index")
    plt.ylabel("Score")
    plt.legend()
    
    # Save the plot with the model name
    plt.tight_layout()
    plt.savefig(f'results/sorted_scores_{title}.png')
    plt.close()

def plot_logits_softmax(model, sample_image, device, title=""):
    """Plot logits and softmax distributions for a sample image."""
    ensure_results_dir()
    
    # Plot the input image
    plt.figure(figsize=(12,4))
    plt.subplot(1, 3, 1)
    img = sample_image.cpu()
    
    # Denormalize the image
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    img = img * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(f"Input Image")
    plt.axis('off')
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        logits = model(sample_image.unsqueeze(0).to(device))
        softmax = F.softmax(logits, dim=1)
        
        logits = logits.cpu().detach().numpy().flatten()
        softmax = softmax.cpu().detach().numpy().flatten()
        
        # Plot logits
        plt.subplot(1, 3, 2)
        plt.bar(np.arange(len(logits)), logits)
        plt.title(f"Logits - {title}")
        plt.xlabel("Class")
        plt.ylabel("Logit Value")
        
        # Plot softmax
        plt.subplot(1, 3, 3)
        plt.bar(np.arange(len(softmax)), softmax)
        plt.title(f"Softmax Output - {title}")
        plt.xlabel("Class")
        plt.ylabel("Probability")
        
        # Convert logits to tensor before using argmax
        logits_tensor = torch.tensor(logits)
        predicted_class = torch.argmax(logits_tensor).item()
        confidence = softmax[predicted_class]
        print(f"\nPredicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        
        # Save the plot with the model name
        plt.tight_layout()
        plt.savefig(f'results/prediction_analysis_{title}.png')
        plt.close()

def plot_roc_pr_curves(scores_id, scores_ood, title=""):
    """
    Plot ROC and PR curves for OOD detection.
    
    Args:
        scores_id: Confidence scores for in-distribution data
        scores_ood: Confidence scores for out-of-distribution data
        model_name: The name of the model
    """
    ensure_results_dir()
    
    # Prepare labels (1 for ID, 0 for OOD)
    y_true = np.concatenate([np.ones(len(scores_id)), np.zeros(len(scores_ood))])
    # Combine scores
    scores = np.concatenate([scores_id.cpu().numpy(), scores_ood.cpu().numpy()])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot ROC curve
    RocCurveDisplay.from_predictions(
        y_true,
        scores,
        ax=ax1,
        name="OOD Detection"
    )
    ax1.set_title(f"Receiver Operating Characteristic (ROC) Curve - {title}")
    
    # Plot Precision-Recall curve
    PrecisionRecallDisplay.from_predictions(
        y_true,
        scores,
        ax=ax2,
        name="OOD Detection"
    )
    ax2.set_title(f"Precision-Recall Curve - {title}")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'results/roc_pr_curves_{title}.png')
    plt.close()

def calculate_metrics(scores_id, scores_ood):
    """
    Calculate various metrics for OOD detection performance.
    
    Args:
        scores_id: Confidence scores for in-distribution data
        scores_ood: Confidence scores for out-of-distribution data
        
    Returns:
        dict: Dictionary containing various metrics
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn import metrics

    # Prepare labels and scores
    y_true = np.concatenate([np.ones(len(scores_id)), np.zeros(len(scores_ood))])
    scores = np.concatenate([scores_id.cpu().numpy(), scores_ood.cpu().numpy()])
    
    # Calculate metrics
    fpr, tpr, _ = metrics.roc_curve(y_true
    , scores)
    tpr_95_index = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95_tpr = fpr[tpr_95_index]
    auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(y_true, scores)
    auroc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)
    
    metrics = {
        'AUC': auc,
        'FPR at 95% TPR': fpr_at_95_tpr,
        'AUROC': auroc,
        'AUPR': aupr
    }
    
    # Print metrics
    print(f"\nOOD Detection Metrics:\n{metrics}")
    
    return metrics

def save_model(model, epoch, accuracy, filename):
    """Save the trained model along with training info."""
    ensure_results_dir()
    save_dict = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy,
    }
    torch.save(save_dict, filename)

def load_model(model, filename, device):
    """
    Load a saved model.
    
    Args:
        model: The model architecture to load weights into
        filename: Path to the saved model file
        device: Device to load the model on
    
    Returns:
        model: The loaded model
        metadata: Dictionary containing training metadata
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No model file found at {filename}")
        
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Print model info
    print(f"\nLoaded model from {filename}")
    print(f"Training epochs: {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['accuracy']:.2f}%")
    
    return model, checkpoint
