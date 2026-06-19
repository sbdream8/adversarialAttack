import torch
import torch.nn.functional as F

def pgd_attack(model, images, labels, epsilon, alpha, iters, device):

    images = images.clone().detach().to(device)
    labels = labels.to(device)
    ori_images = images.clone().detach()

    for _ in range(iters):

        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = (images + alpha * images.grad.sign())
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, 0, 1).detach()

    return images