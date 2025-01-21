import torch
import torch.nn.functional as F
import os 

def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs('results', exist_ok=True)

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