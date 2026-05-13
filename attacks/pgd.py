import torch
import torch.nn.functional as F

def pgd_attack(
    model,
    images,
    labels,
    epsilon=0.03,
    alpha=0.007,
    iters=10,
    device="cuda"
):

    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.data
        adv_images = images + alpha * grad.sign()
        eta = torch.clamp(
            adv_images - original_images,
            min=-epsilon,
            max=epsilon
        )

        images = torch.clamp(
            original_images + eta,
            0,
            1
        ).detach()

    return images