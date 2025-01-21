import torch
from sklearn import metrics
import argparse
from models.cnn import CNN 
import os
from odin import Odin
import argparse
import torch
import torchvision
from torchvision.datasets import FakeData
from torch.utils.data import Subset
from torchvision import transforms
import os
from models.cnn import CNN 
from utils import load_model
from models import get_model

def grid_search(model, testloader, fakeloader, temperatures, epsilons, device='cpu'):
    model.eval()
    odin = Odin(model)
    best_auc = 0
    best_temp = None
    best_eps = None
    for temp in temperatures:
        for eps in epsilons:
            odin.set_temperature(temp)
            odin.set_epsilon(eps)
            scores_test = odin(testloader)
            scores_fake = odin(fakeloader)
            prediction = torch.cat((scores_test, scores_fake))
            target = torch.cat((torch.ones_like(scores_test), torch.zeros_like(scores_fake)))
            fpr, tpr, _ = metrics.roc_curve(target.cpu().numpy(), prediction.cpu().numpy())
            auc_score = metrics.auc(fpr, tpr)
            if auc_score > best_auc:
                print(f'Temperature: {temp}, Epsilon: {eps}, AUC: {auc_score}')
                best_auc = auc_score
                best_temp = temp
                best_eps = eps
    return best_temp, best_eps, best_auc


def parse_args():
    parser = argparse.ArgumentParser(description="OOD Detection Baseline")
    parser.add_argument('--model-type', type=str, choices=['cnn', 'resnet'],
                       default='cnn', help='Type of model to use (default: cnn)')
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and evaluation")
    parser.add_argument("--ood-set", type=str, default="fakedata", choices=["fakedata", "cifar100"],
                        help="OOD dataset to evaluate")
    parser.add_argument('--model-path', type=str, 
                       help='Path to the model file')
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print("Using CPU")
        
    print(f"\nInitializing {args.model_type.upper()} model...")
    model = get_model(args.model_type).to(device)
    
    print("\nLoading the pretrained model...")
    try:
        model, checkpoint = load_model(model, args.model_path, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using --train flag or provide correct model path")
        
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # Settings
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2
    id_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # OOD dataset selection
    if args.ood_set == 'fakedata':
        fakeset = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
        fakeloader = torch.utils.data.DataLoader(fakeset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if args.ood_set == 'cifar100':
        ood_classes = ["aquarium", "bycicle", "bottle", "bed", "rocket", "can", "girl", "chair"]
        cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        selected_indices = [i for i, (_, label) in enumerate(cifar100) if cifar100.classes[label] in ood_classes]
        fakeset = Subset(cifar100, selected_indices)
        fakeloader = torch.utils.data.DataLoader(fakeset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    os.makedirs('./results', exist_ok=True)
    dir = os.path.splitext(os.path.basename(args.model_type))[0]
    os.makedirs(f'./results/{dir}', exist_ok=True)
    
    temperatures = [1, 10, 20, 50, 100, 200, 500]
    epsilons = [0.01, 0.02, 0.03, 0.04]

    best_temp, best_eps, best_auc = grid_search(model, testloader, fakeloader, temperatures, epsilons, device=device)
    print(f'Best Temp: {best_temp}')
    print(f'Best Eps: {best_eps}')
    print(f'Best AUC: {best_auc}')













