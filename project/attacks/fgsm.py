import torch
import torch.nn.functional as F

class fgsm_attack:

    def __init__(self, model, epsilon, device):
        self.model = model
        self.epsilon = epsilon
        self.device = device

    def perturb(self, images, labels):

        images = (images.clone().detach().to(self.device))
        labels = labels.to(self.device)
        images.requires_grad = True
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        grad = images.grad.data
        adv_images = (images + self.epsilon * grad.sign())

        return adv_images.detach()