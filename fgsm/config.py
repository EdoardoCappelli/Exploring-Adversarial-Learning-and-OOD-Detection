import torch

CONFIG = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_workers': 4 if torch.cuda.is_available() else 2,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'epsilon': 0.01,
    'max_iter': 100,
    'train_epochs': 10,
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'transform_stats': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    }
}

