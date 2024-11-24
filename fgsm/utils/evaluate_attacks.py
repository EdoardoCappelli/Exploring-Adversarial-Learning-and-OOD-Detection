import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Subset
import random
import matplotlib.pyplot as plt
from attack.fgsm import fgsm_attack
from config import CONFIG
from models.cnn import CNN

class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def evaluate_attack_success_rate(model, testloader, criterion, epsilon, max_iter, device):
    """Evaluates attack success rate for given parameters"""
    num_attacks = 0
    num_success = 0
    
    for x, y in testloader:
        for img, label in zip(x, y):
            img, label = img.to(device), label.to(device)
            num_iter, _, _, _ = fgsm_attack(model, criterion, img, label, epsilon, max_iter)
            
            if num_iter != 0:
                num_attacks += 1
                if num_iter is not None:
                    num_success += 1
                    
    return num_success / num_attacks if num_attacks > 0 else 0

def main():
    # Load model and dataset
    model = CNN().to(CONFIG['device'])
    model.load_state_dict(torch.load('checkpoints/cifar10_CNN_adv_10.pth', weights_only=True))
    model.eval()
    
    transform = torchvision.transforms.Compose([  
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**CONFIG['transform_stats'])
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # EVALUATE ASR VERSUS EPSILON
    epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    batch_size = 256
    model.eval()
    criterion = nn.CrossEntropyLoss()
    inv = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    asr_eps = []  # Initialize list to store Attack Success Rate for each epsilon
    for epsilon in epsilons:
        num_attacks = 0
        num_success = 0
        subset = Subset(testset, random.sample(range(len(testset)), 1000))
        subloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=CONFIG['num_workers'])
        for x, y in subloader:
            for img, label in zip(x, y):
                img, label = img.to(CONFIG['device']), label.to(CONFIG['device'])
                num_iter, orig_img, adv_img, pred = fgsm_attack(model, criterion, img, label, epsilon, max_iter=1)
                if num_iter != 0:
                    num_attacks += 1
                    if num_iter is not None:
                        num_success += 1
        asr_eps.append(num_success / num_attacks)  # Store ASR for current epsilon
        print(f'Epsilon: {epsilon} - Attack Success Rate: {round(num_success / num_attacks, 2)}')
        
    # EVALUATE ASR VERSUS MAX ITERATIONS
    epsilon = 0.001
    max_iters = [1, 5, 10, 20, 50]
    batch_size = 256
    model.eval()
    criterion = nn.CrossEntropyLoss()
    inv = NormalizeInverse((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    asr_iters = []  # Initialize list to store Attack Success Rate for each max iteration
    for max_iter in max_iters:
        num_attacks = 0
        num_success = 0
        subset = Subset(testset, random.sample(range(len(testset)), 1000))
        subloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=CONFIG['num_workers'])
        for x, y in subloader:
            for img, label in zip(x, y):
                img, label = img.to(CONFIG['device']), label.to(CONFIG['device'])
                num_iter, orig_img, adv_img, pred = fgsm_attack(model, criterion, img, label, epsilon, max_iter=max_iter)
                if num_iter != 0:
                    num_attacks += 1
                    if num_iter is not None:
                        num_success += 1
        asr_iters.append(num_success / num_attacks)  # Store ASR for current max iteration
        print(f'Max Iterations: {max_iter} - Attack Success Rate: {round(num_success / num_attacks, 2)}')
        
        
    plt.figure()
    plt.plot(epsilons, asr_eps, label='CIFAR-10 Training')
    plt.xlabel('Epsilon')
    plt.ylabel('Attack Success Rate')
    plt.title('Attack Success Rate vs Epsilon')
    plt.legend()
    plt.savefig('results/ASR_epsilon.png')
    plt.show()
    
    plt.figure()
    plt.plot(max_iters, asr_iters, label='CIFAR-10 Training')
    plt.xlabel('Max Iterations')
    plt.ylabel('Attack Success Rate')
    plt.title('Attack Success Rate vs Max Iterations')
    plt.legend()
    plt.savefig('results/ASR_maxiter.png')
    plt.show()

if __name__ == "__main__":
    main()
