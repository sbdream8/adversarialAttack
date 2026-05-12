import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from tqdm import tqdm
from models.resnet import ResNet18
from models.cnn import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


EPOCHS = 50
LR = 0.1

#나중에 attack할때 헷갈릴 수 있어서 model1, model2 하는것보단 dictionary 형태 추천
models = {
    "ResNet18": ResNet18().to(device),
    "SimpleCNN": SimpleCNN().to(device)
}

optimizers = {
    name: optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=5e-4
    )
    for name, model in models.items()
}

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizers,
    milestones=[25, 40],
    gamma=0.1
)

criterion = nn.CrossEntropyLoss()

for name, model in models.items():
    optimizer = optimizers[name]
    print(f"\nTraining {name}")

    for epoch in range(EPOCHS):
        train_loss = train(
            model,
            train_loader,
            optimizer,
            criterion
        )
        test_acc = evaluate(
            model,
            test_loader
        )
        print(
            f"{name} | "
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Acc: {test_acc:.2f}%"
        )

    torch.save(
        model.state_dict(),
        f"./checkpoints/{name}.pth"
    )