import torch
import torchvision
from torchvision.datasets import FakeData
from torch.utils.data import Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import argparse
from models.cnn import CNN 
import os
from odin import Odin

def parse_args():
    parser = argparse.ArgumentParser(description="OOD Detection Baseline")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and evaluation")
    parser.add_argument('--epsilon', type=float, default=0.01, help='perturbation')
    parser.add_argument('--temp', type=float, default=0.01, help='temperature')
    parser.add_argument("--ood_set", type=str, default="fakedata", choices=["fakedata", "cifar100"],
                        help="OOD dataset to evaluate")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to the pretrained model"),
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(args.pretrained, weights_only=True))
    model.eval()

    os.makedirs('./results', exist_ok=True)
    dir = os.path.splitext(os.path.basename(args.pretrained))[0]
    os.makedirs(f'./results/{dir}', exist_ok=True)
    
    
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
   
    temp = args.temp 
    eps = args.epsilon
    odin = Odin(model, temperature=temp, epsilon=eps)
    scores_test_odin = odin(testloader)
    scores_fake_odin = odin(fakeloader)

    prediction = torch.cat((scores_test_odin, scores_fake_odin))
    target = torch.cat((torch.ones_like(scores_test_odin), torch.zeros_like(scores_fake_odin)))
    
    fpr_odin, tpr_odin, _ = metrics.roc_curve(target.cpu().numpy(), prediction.cpu().numpy())
    auc_score_odin = metrics.auc(fpr_odin, tpr_odin)
    print(f'AUC ROC (ODIN): {auc_score_odin}')
    tpr_95_index = np.argmin(np.abs(tpr_odin - 0.95))
    fpr_at_95_tpr_odin = fpr_odin[tpr_95_index]
    print(f'FPR at 95% TPR (ODIN): {fpr_at_95_tpr_odin}')
    metrics.RocCurveDisplay.from_predictions(target.cpu().numpy(), prediction.cpu().numpy())

    precision_odin, recall_odin, _ = metrics.precision_recall_curve(target.cpu().numpy(), prediction.cpu().numpy())
    print(f'AUC PRC: {metrics.auc(recall_odin, precision_odin)}')
    print(f'Precision: {precision_odin[recall_odin.argmax()]}')
    print(f'Recall: {recall_odin[recall_odin.argmax()]}')
    metrics.PrecisionRecallDisplay.from_predictions(target.cpu().numpy(), prediction.cpu().numpy())

    plt.figure()
    plt.plot(fpr_odin, tpr_odin, label='ODIN')
    plt.legend()
    plt.savefig(f'results/{dir}/frp_tpr_odin.png')
    if args.verbose:
        plt.show()
    
    plt.figure()
    plt.plot(precision_odin, recall_odin, label='ODIN')
    plt.legend()
    plt.savefig(f'results/{dir}/precision_recall_odin.png')
    if args.verbose:
        plt.show()