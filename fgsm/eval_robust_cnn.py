import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import argparse
from utils import NormalizeInverse
from models.cnn import CNN 
import os
import random 
from fgsm import fgsm_attack

def parse_args():
    parser = argparse.ArgumentParser(description="OOD Detection Baseline")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and evaluation")
    parser.add_argument("--ood_set", type=str, default="fakedata", choices=["fakedata", "cifar100"],
                        help="OOD dataset to evaluate")
    parser.add_argument("--robust-model", type=str, required=True, help="Path to the robust model"),
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    num_workers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(args.robust_model, weights_only=True))
    model.eval()

    os.makedirs('./results', exist_ok=True)
    dir = os.path.splitext(os.path.basename(args.robust_model))[0]
    os.makedirs(f'./results/{dir}', exist_ok=True)
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    # EVALUATE ASR VERSUS EPSILON
    # epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    epsilons = [0.001, 0.005,]
    model.eval()
    criterion = nn.CrossEntropyLoss()
    inv = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    asr_eps_adv = []
    for epsilon in epsilons:
        num_attacks = 0
        num_success = 0
        subset = Subset(testset, random.sample(range(len(testset)), 1000))
        subloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for x, y in subloader:
            for img, label in zip(x, y):
                img, label = img.to(device), label.to(device)
                num_iter, orig_img, adv_img, pred = fgsm_attack(model, criterion, img, label, epsilon, max_iter=1)
                if num_iter != 0:
                    num_attacks += 1
                    if num_iter != None:
                        num_success += 1
        asr_eps_adv.append(num_success/num_attacks)
        print(f'Epsilon: {epsilon} - Attack Success Rate: {num_success/num_attacks}')
        
    #EVALUATE ASR VERSUS MAX ITERATIONS
    epsilon = 0.001
    # max_iters = [1, 5, 10, 20, 50]
    max_iters = [1, 5]
    model.eval()
    criterion = nn.CrossEntropyLoss()
    inv = NormalizeInverse((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    asr_iters_adv = []
    for max_iter in max_iters:
        num_attacks = 0
        num_success = 0
        subset = Subset(testset, random.sample(range(len(testset)), 1000))
        subloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for x, y in subloader:
            for img, label in zip(x, y):
                img, label = img.to(device), label.to(device)
                num_iter, orig_img, adv_img, pred = fgsm_attack(model, criterion, img, label, epsilon, max_iter=max_iter)
                if num_iter != 0:
                    num_attacks += 1
                    if num_iter != None:
                        num_success += 1
        asr_iters_adv.append(num_success/num_attacks)
        print(f'Max Iterations: {max_iter} - Attack Success Rate: {num_success/num_attacks}')
            
    plt.figure()
    plt.plot(epsilons, asr_eps_adv, label='Adversarial Training')
    plt.xlabel('Epsilon')
    plt.ylabel('Attack Success Rate')
    plt.title('Attack Success Rate vs Epsilon')
    plt.legend()
    plt.savefig(f'results/{dir}/ASR_epsilon.png')
    if args.verbose:
        plt.show()
        
    plt.figure()
    plt.plot(max_iters, asr_iters_adv, label='Adversarial Training')
    plt.xlabel('Max Iterations')
    plt.ylabel('Attack Success Rate')
    plt.title('Attack Success Rate vs Max Iterations')
    plt.legend()
    
    plt.savefig(f'results/{dir}/ASR_max_iter.png')
    if args.verbose:
        plt.show()
        
