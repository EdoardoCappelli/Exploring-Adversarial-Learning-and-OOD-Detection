import torch
import torch.nn as nn

def fgsm_attack(model, criterion, image, label, epsilon, max_iter=100):
    """
    Performs FGSM attack on a single image
    
    Args:
        model: target model
        criterion: loss function
        image: input image
        label: true label
        epsilon: attack strength
        max_iter: maximum number of iterations
        
    Returns:
        num_iter: number of iterations needed (None if failed)
        orig_img: original image
        perturb_img: perturbed image
        pred: prediction on perturbed image
    """
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
