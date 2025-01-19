import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loaders(batch_size=256):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    num_workers = 2
    
    # CIFAR-10 datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,                  
                                           download=True, transform=transform)

    # CIFAR-100 for OOD data
    cifar100 = torchvision.datasets.CIFAR100(root='./data', train=False, 
                                           download=True, transform=transform)
    
    ood_classes = ["maple_tree", "aquarium_fish", "willow_tree", "flatfish", "rose", "lawn_mower", "porcupine", "caterpillar", "seaweed", "shrew"]
    selected_indices = [i for i, (_, label) in enumerate(cifar100) if cifar100.classes[label] in ood_classes]
    ood_subset = torch.utils.data.Subset(cifar100, selected_indices)

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                            shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                           shuffle=False, num_workers=num_workers)
    oodloader = torch.utils.data.DataLoader(ood_subset, batch_size=batch_size, 
                                          shuffle=False, num_workers=num_workers)

    return trainloader, testloader, oodloader
