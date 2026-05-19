import torch
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon, device):

    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    adv_images = (images + epsilon * images.grad.sign())
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()