import torch
import torch.nn.functional as F

def pgd_attack(model, images, labels, epsilon, alpha, iters, device):

    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1,3,1,1)
    eps = epsilon / std
    alp = alpha / std

    images = images.clone().detach().to(device)
    labels = labels.to(device)
    ori_images = images.clone().detach()

    for _ in range(iters):

        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad( loss, images, retain_graph=False, create_graph=False )[0]
        model.zero_grad()
        loss.backward()
        adv_images = (images + alpha * images.grad.sign())
        # eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon) ->
        eta = torch.max( torch.min( adv_images - ori_images, eps ), -eps )
        # images = torch.clamp(ori_images + eta, 0, 1).detach() ->
        images = (ori_images + eta).detach()

    return images