import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from train import (DEVICE, MODEL_DICT, get_dataloaders, CHECKPOINT_DIR)
from defenses.trades import trades_loss
from defenses.arow import arow_loss


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "cnn"])
parser.add_argument("--defense", type=str, default="trades", choices=["trades", "arow"])
parser.add_argument("--epochs", type=int, default=50)
args = parser.parse_args()

def main():

    train_loader, test_loader = get_dataloaders()
    model = MODEL_DICT[args.model]().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)

    best_acc = 0

    for epoch in range(args.epochs):

        model.train()
        total_loss = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if args.defense == "trades":

                loss = trades_loss(model, images, labels, optimizer)

            elif args.defense == "arow":

                loss = arow_loss(model, images, labels, optimizer)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss: {total_loss/len(train_loader):.4f}"
        )

        torch.save(
            model.state_dict(),
            CHECKPOINT_DIR /
            f"{args.model}_{args.defense}.pth"
        )


if __name__ == "__main__":
    main()