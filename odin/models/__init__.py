from .cnn import CNN
from .resnet import ResNet

def get_model(model_name):
    """
    Factory function to get the specified model.
    
    Args:
        model_name: str, either 'cnn' or 'resnet'
    
    Returns:
        A model instance
    """
    models = {
        'cnn': CNN,
        'resnet': ResNet
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not recognized. Choose from: {list(models.keys())}")
    
    return models[model_name]()