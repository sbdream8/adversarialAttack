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
    
    clean_loss = F.cross_entropy(logits_clean, y, label_smoothing=0.1)
    prob_clean = F.softmax(logits_clean, dim=1)
    log_prob_adv = F.log_softmax(logits_adv, dim=1)
    kl = F.kl_div(log_prob_adv, prob_clean, reduction='none').sum(dim=1)
    prob_adv = F.softmax(logits_adv, dim=1)
    p_y_adv = prob_adv.gather(1, y.unsqueeze(1)).squeeze(1)
    weight = 1.0 - p_y_adv
    reg_loss = (kl * weight).mean()
    loss = clean_loss + beta * reg_loss

    return loss