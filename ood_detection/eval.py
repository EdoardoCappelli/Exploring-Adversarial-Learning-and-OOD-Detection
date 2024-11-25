import torch
import torchvision
from torchvision.datasets import FakeData
from torch.utils.data import Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn import metrics
import argparse
from utils import evaluate_model
from models.cnn import CNN 
import os

def max_logit(logit):
    s, _ = logit.max(dim=1) #get the max for each element of the batch
    return s

# BASELINE: https://arxiv.org/pdf/1610.02136
def max_softmax(logit, T=1.0):
    s = F.softmax(logit/T, 1)
    s, _ = s.max(dim=1) #get the max for each element of the batch
    return s

def compute_scores(data_loader, score_function):
    scores = []
    with torch.no_grad():
        for data in data_loader:
            x, y = data
            output = model(x.to(device))
            s = score_function(output)
            scores.append(s.cpu())
        scores_t = torch.cat(scores)
        return scores_t

def parse_args():
    parser = argparse.ArgumentParser(description="OOD Detection Baseline")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and evaluation")
    parser.add_argument("--ood_set", type=str, default="fakedata", choices=["fakedata", "cifar100"],
                        help="OOD dataset to evaluate")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to the pretrained model"),
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    batch_size = args.batch_size # default 256
    num_workers = 2

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if args.ood_set == 'fakedata':
      fakeset = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
      fakeloader = torch.utils.data.DataLoader(fakeset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if args.ood_set == 'cifar100':
      ood_classes = ["aquarium", "bycicle", "bottle", "bed", "rocket", "can", "girl", "chair"]
      cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
      selected_indices = [i for i, (_, label) in enumerate(cifar100) if cifar100.classes[label] in ood_classes]
      fakeset = Subset(cifar100, selected_indices)
      fakeloader = torch.utils.data.DataLoader(fakeset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(args.pretrained, weights_only=True))

    os.makedirs('./results', exist_ok=True)
    dir = os.path.splitext(os.path.basename(args.pretrained))[0]
    os.makedirs(f'./results/{dir}', exist_ok=True)
    
    # ACCURACY
    loss =  nn.CrossEntropyLoss()
    _, test_acc, report = evaluate_model(model, testloader, loss, device=device)
    if args.verbose:
        print(f'{args.pretrained} test accuracy: {test_acc}')
        print(report)
    fig, ax = plt.subplots(figsize=(10, 8))  # Imposta una dimensione adatta
    ax.axis('off')  # Disattiva gli assi
    ax.text(0.1, 0.5, report, fontsize=12, ha='left', va='center', wrap=True)
    plt.savefig(f'results/{dir}/classification_report.png', bbox_inches='tight')
    plt.close()

    # LOGIT AND SOFTMAX OUTPUT WITH ID DATA
    x, y = next(iter(testloader))
    x, y = x.to(device), y.to(device)
    output = model(x)
    plt.bar(np.arange(10),output[0].detach().cpu())
    plt.title('logit (ID Data)')
    plt.savefig(f'results/{dir}/logit_id_data.png')
    if args.verbose:
        plt.show()
    
    T=1
    plt.title(f'softmax t={T} (ID Data)')
    s = F.softmax(output/T, 1)
    plt.bar(np.arange(10),s[0].detach().cpu())
    plt.savefig(f"results/{dir}/softmax_T_{T}_id_data.png")
    if args.verbose:
        plt.show()

    # BASELINE FOR DETECTING MISCLASSIFIED AND OUT-OF-DISTRIBUTION EXAMPLES IN NEURAL NETWORKS
    score_function = max_softmax
    scores_test = compute_scores(testloader, score_function)
    scores_fake = compute_scores(fakeloader, score_function)

    plt.figure()
    plt.plot(sorted(scores_test), label='ID')
    plt.plot(sorted(scores_fake), label='OOD')
    plt.legend()
    plt.savefig(f"results/{dir}/baseline_plot_id_vs_{args.ood_set}.png")

    plt.figure()
    plt.hist(scores_test, density=True, alpha=0.5, bins=25, label='ID')
    plt.hist(scores_fake, density=True, alpha=0.5, bins=25, label='OOD')
    plt.legend()
    plt.savefig(f"results/{dir}/baseline_hist_id_vs_{args.ood_set}.png")
    if args.verbose:
        plt.show()

    # EVALUATE OOD DETECTION 
    prediction = torch.cat((scores_test, scores_fake))
    target = torch.cat((torch.ones_like(scores_test), torch.zeros_like(scores_fake)))

    fpr, tpr, _ = metrics.roc_curve(target.cpu().numpy(), prediction.cpu().numpy())
    auc_score = metrics.auc(fpr, tpr)
    tpr_95_index = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95_tpr = fpr[tpr_95_index]
    metrics.RocCurveDisplay.from_predictions(target.cpu().numpy(), prediction.cpu().numpy())
    plt.savefig(f"results/{dir}/roc_curve_id_{args.ood_set}.png")

    precision, recall, _ = metrics.precision_recall_curve(target.cpu().numpy(), prediction.cpu().numpy())
    auc_prc = metrics.auc(recall, precision)
    precision_value = precision[recall.argmax()]
    recall_value = recall[recall.argmax()]
    
    metrics.PrecisionRecallDisplay.from_predictions(target.cpu().numpy(), prediction.cpu().numpy())
    plt.savefig(f"results/{dir}/prc_curve_id_{args.ood_set}.png")
    
    if args.verbose:
        print(f'AUC PRC: {auc_prc}')
        print(f'Precision: {precision_value}')
        print(f'Recall: {recall_value}')
        print(f'AUC ROC: {auc_score}')
        print(f'FPR at 95% TPR: {fpr_at_95_tpr}')
    
    fig, ax = plt.subplots(figsize=(8, 4))  # Imposta la dimensione della figura
    ax.axis('off')
    text = f'AUC PRC: {auc_prc:.4f}\nPrecision: {precision_value:.4f}\nRecall: {recall_value:.4f}\nAUC ROC: {auc_score:.4f}\nFPR at 95% TPR: {fpr_at_95_tpr:.4f}'
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, wrap=True)
    plt.savefig(f"results/{dir}/precision_recall_info_{args.ood_set}.png", bbox_inches='tight')
    plt.close()


