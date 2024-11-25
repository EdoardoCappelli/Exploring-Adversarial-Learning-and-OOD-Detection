import argparse
import torch
import torchvision
from torchvision.datasets import FakeData
from torch.utils.data import Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import random
import gc
import os
from utils import NormalizeInverse
from models.cnn import CNN 

def fgsm_attack(model, criterion, image, label, epsilon, max_iter=100):
    orig_img = image.clone().detach()
    perturb_img = image.clone().detach().requires_grad_(True)

    output = model(perturb_img.unsqueeze(0))
    if output.argmax().item() != label.item():
        return 0, orig_img, perturb_img.detach(), output.argmax().item()
    for i in range(max_iter + 1):
        output = model(perturb_img.unsqueeze(0))
        loss = criterion(output, label.unsqueeze(0))
        pred = output.argmax().item()
        if pred != label.item():
            return i, orig_img, perturb_img.detach(), pred

        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            perturb_img += epsilon * torch.sign(perturb_img.grad)

        perturb_img.requires_grad_(True)

    return None, orig_img, perturb_img.detach(), pred


def show_attack(orig_img, label, adv_img, pred, num_iter, inv, classes, i, verbose):
    if num_iter == None:
        print('Attack Failed!')
    elif num_iter == 0:
        print('Already misclassified!')
    else:
        print(f'Attack Success!')
        print(f'Iteration: {num_iter}')
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(inv(orig_img).permute(1,2,0).detach().cpu())
        ax[0].set_title(f'Original - {classes[label]}')
        ax[1].imshow(inv(adv_img).permute(1,2,0).detach().cpu())
        ax[1].set_title(f'Adversarial - {classes[pred]}')
        ax[2].imshow(inv(orig_img-adv_img).permute(1,2,0).detach().cpu())
        ax[2].set_title('Difference')
        plt.savefig(f'results/{dir}/adv_example_{i}')
        if verbose:
            plt.show()


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR10 with optional OOD dataset")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument("--pretrained", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument('--epsilon', type=float, default=0.01, help='perturbation')
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    epsilon = args.epsilon 
    batch_size = args.batch_size
    criterion = nn.CrossEntropyLoss()
    inv = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_workers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(args.pretrained, weights_only=True))

    os.makedirs('./results', exist_ok=True)
    dir = os.path.splitext(os.path.basename(args.pretrained))[0]
    os.makedirs(f'./results/{dir}', exist_ok=True)
    model.eval()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    img, label = next(iter(testloader))
    for i in range(3):
        img_ex, label_ex = img[i].to(device), label[i].to(device)
        num_iter, orig_img, adv_img, pred = fgsm_attack(model, criterion, img_ex, label_ex, epsilon)
        show_attack(orig_img, label_ex, adv_img, pred, num_iter, inv, classes, i, args.verbose)

























