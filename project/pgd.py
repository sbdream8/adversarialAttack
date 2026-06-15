import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.resnet import ResNet18
from models.cnn import SimpleCNN

from torchvision import datasets
from torch.utils.data import DataLoader

transform_test = transforms.ToTensor()

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
CHECKPOINT_DIR = "/content/drive/MyDrive/adversarialAttack/adversarial_checkpoints"
cnn = SimpleCNN().to(device)
cnn.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/cnn_best.pth", map_location=device))

resnet = ResNet18().to(device)
resnet.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/resnet18_best.pth", map_location=device))

def evaluate_clean(model, test_loader, device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            pred = outputs.argmax(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

def pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, num_steps=10, device="cuda"):

    images = images.clone().detach().to(device)
    labels = labels.to(device)

    adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1)

    for _ in range(num_steps):

        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()

        # Projection
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)

        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images


def evaluate_pgd(model, test_loader, device):

    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        adv_images = pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, num_steps=10, device=device)
        outputs = model(adv_images)
        pred = outputs.argmax(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return 100 * correct / total

if __name__ == "__main__":

    print("Loading CIFAR10...")

    clean_acc = evaluate_clean(cnn, test_loader, device)
    pgd_acc = evaluate_pgd(cnn, test_loader, device)

    print(f"CNN Clean Acc : {clean_acc:.2f}%")
    print(f"CNN PGD Acc   : {pgd_acc:.2f}%")