import torch

CONFIG = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_workers': 4 if torch.cuda.is_available() else 2,
    'batch_size': 256,
    'learning_rate': 0.0001,
    'train_epochs': [10],
    'transform_stats': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    },
    'checkpoint_dir': './checkpoints',
    'data_dir': './data',
    'data_loader': 'cifar100', # 'fakedata'
}