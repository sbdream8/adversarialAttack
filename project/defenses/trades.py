import torch
import torch.nn.functional as F
from torchgen import model


def trades_loss(model, x_natural, y, optimizer, step_size=2/255, epsilon=8/255, perturb_steps=10, beta=6.0):

    criterion_kl = torch.nn.KLDivLoss(reduction="batchmean")
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)

    with torch.no_grad():
        natural_probs = F.softmax(model(x_natural), dim=1)

    for _ in range(perturb_steps):

        x_adv.requires_grad_()

        with torch.enable_grad():

            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), natural_probs)

        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)

    model.train()
    optimizer.zero_grad()
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust

    print("loss_natural nan:", torch.isnan(loss_natural))
    print("loss_robust nan:", torch.isnan(loss_robust))
    print("total loss nan:", torch.isnan(loss))

    return loss