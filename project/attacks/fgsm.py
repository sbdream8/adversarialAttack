import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon, device):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad = images.grad.data
    adv_images = images + epsilon * grad.sign()

    return adv_images.detach()