import torch
import torch.nn.functional as F

class pgd_attack:

    def __init__(self, model, epsilon, alpha, steps, device):

        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.device = device

    def perturb(self, images, labels):

        images = (images.clone().detach().to(self.device))
        labels = labels.to(self.device)
        original_images = (images.clone().detach())

        # Random Start
        images = (images + torch.empty_like(images).uniform_(-self.epsilon,self.epsilon))

        for _ in range(self.steps):

            images.requires_grad = True
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, labels)
            self.model.zero_grad()
            loss.backward()
            grad = images.grad.data
            adv_images = (images + self.alpha * grad.sign())
            eta = torch.clamp(adv_images - original_images, min=-self.epsilon, max=self.epsilon)
            images = (original_images + eta).detach()

        return images