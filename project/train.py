# Train Script for CIFAR10 Classification using CNN / ResNet

import random
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from models.resnet import ResNet18
from models.cnn import SimpleCNN


# Argument Parser
parser = argparse.ArgumentParser(description="Train CNN / ResNet on CIFAR10")
parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "cnn"])
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
print(args)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

torch.backends.cudnn.benchmark = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Checkpoint Directory
CHECKPOINT_DIR = Path("/content/drive/MyDrive/adversarial_checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

cifar10_mean = (0.4914, 0.4822, 0.4465)

cifar10_std = (0.2470, 0.2435, 0.2616)


def get_dataloaders():

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

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

# Models
MODEL_DICT = {
    "resnet18": ResNet18,
    "cnn": SimpleCNN
}


# Train
def train_one_epoch(model, loader, optimizer, criterion):

    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# Evaluation
def evaluate(model, loader):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# Main
def main():

    set_seed(args.seed)
    train_loader, test_loader = (get_dataloaders())
    model = MODEL_DICT[args.model]().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1))
    criterion = nn.CrossEntropyLoss()

    latest_checkpoint_path = (
        CHECKPOINT_DIR /
        f"{args.model}_latest.pth"
    )

    best_checkpoint_path = (
        CHECKPOINT_DIR /
        f"{args.model}_best.pth"
    )

    start_epoch = 0
    best_acc = 0

    # Resume Checkpoint
    if latest_checkpoint_path.exists():

        print(
            f"Loading checkpoint "
            f"for {args.model}"
        )

        checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_acc = checkpoint["best_acc"]
        start_epoch = (checkpoint["epoch"] + 1)

        print(
            f"Resume from epoch "
            f"{start_epoch}"
        )


    # Training Loop
    for epoch in range(start_epoch, args.epochs):

        train_loss = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion)
        test_acc = evaluate(model, test_loader)
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{args.epochs}] | "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {test_acc:.2f}%"
        )

        # Save Latest
        torch.save({
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state_dict": (model.state_dict()),
            "optimizer_state_dict": (optimizer.state_dict()),
            "scheduler_state_dict": (scheduler.state_dict())},
            latest_checkpoint_path)

        # Save Best
        if test_acc > best_acc:

            best_acc = test_acc
            torch.save(model.state_dict(),best_checkpoint_path)

            print(
                f"Best Model Saved! "
                f"Acc: {best_acc:.2f}%"
            )


if __name__ == "__main__":
    main()