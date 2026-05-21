import torch
import torch.nn.functional as F


def arow_loss(model, x_natural, y, optimizer, step_size=2/255, epsilon=8/255, perturb_steps=10, beta=6.0):

    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)

    for _ in range(perturb_steps):

        x_adv.requires_grad_()

        with torch.enable_grad():

            logits_adv = model(x_adv)
            loss_adv = F.cross_entropy(logits_adv, y)

        grad = torch.autograd.grad(loss_adv, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)

        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    optimizer.zero_grad()
    logits_clean = model(x_natural)
    logits_adv = model(x_adv)
    clean_loss = F.cross_entropy(logits_clean, y)
    robust_loss = F.cross_entropy(logits_adv, y)
    confidence = torch.softmax(logits_clean, dim=1).max(dim=1)[0]
    weight = (1.0 - confidence).detach()
    robust_loss = (weight * robust_loss).mean()
    loss = clean_loss + beta * robust_loss

    return loss