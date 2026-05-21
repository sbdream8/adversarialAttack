import torch
import torch.nn.functional as F


def trades_loss(model, x_natural, y, optimizer, step_size=2/255, epsilon=8/255, perturb_steps=10, beta=6.0):

    criterion_kl = torch.nn.KLDivLoss(reduction="batchmean")
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)

    for _ in range(perturb_steps):

        x_adv.requires_grad_()

        with torch.enable_grad():

            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1))

        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    optimizer.zero_grad()
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust

    return loss