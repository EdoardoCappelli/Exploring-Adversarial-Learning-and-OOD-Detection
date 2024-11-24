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

def show_attack(orig_img, label, adv_img, pred, num_iter, inv, classes, i):
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
        plt.savefig(f'results/adv_attack_{i}')
        plt.show()
        
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
        
    epsilon = 0.1
    batch_size = 5
    criterion = nn.CrossEntropyLoss()
    inv = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model.eval()

    img, label = next(iter(testloader))
    for i in range(batch_size):
        img_ex, label_ex = img[i].to(CONFIG['device']), label[i].to(CONFIG['device'])
        num_iter, orig_img, adv_img, pred = fgsm_attack(model, criterion, img_ex, label_ex, epsilon)
        show_attack(orig_img, label_ex, adv_img, pred, num_iter, inv, classes, i)

if __name__ == "__main__":
    main()
