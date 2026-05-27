# Adversarial Training

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from train import (DEVICE, MODEL_DICT, get_dataloaders, CHECKPOINT_DIR, evaluate, set_seed)
from defenses.trades import trades_loss
from defenses.arow import arow_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Argument Parser
parser = argparse.ArgumentParser(description="Adversarial Training")
parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "cnn"])
parser.add_argument("--defense", type=str, default="trades", choices=["trades", "arow"])
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()
print(args)


# Main
def main():

    set_seed(args.seed)
    train_loader, test_loader = get_dataloaders(args.batch_size)
    model = MODEL_DICT[args.model]().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)

    latest_checkpoint_path = (
        CHECKPOINT_DIR /
        f"{args.model}_{args.defense}_latest.pth"
    )

    best_checkpoint_path = (
        CHECKPOINT_DIR /
        f"{args.model}_{args.defense}_best.pth"
    )

    start_epoch = 0
    best_acc = 0


    # Resume Checkpoint
    if latest_checkpoint_path.exists():

        print(
            f"Loading checkpoint "
            f"for {args.model} ({args.defense})"
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

        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, labels in progress_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            # Defense Loss
            if args.defense == "trades":

                loss = trades_loss(model=model, x_natural=images, y=labels, optimizer=optimizer)

            elif args.defense == "arow":

                loss = arow_loss(model=model, x_natural=images, y=labels, optimizer=optimizer)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

        scheduler.step()


        # Clean Accuracy
        clean_acc = evaluate(model, test_loader)
        avg_loss = (total_loss / len(train_loader))

        print(
            f"\nEpoch [{epoch+1}/{args.epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Clean Acc: {clean_acc:.2f}%"
        )

        # Save Latest Checkpoint
        torch.save({
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        },
        latest_checkpoint_path)


        # Save Best Model
        if clean_acc > best_acc:

            best_acc = clean_acc
            torch.save(model.state_dict(), best_checkpoint_path)

            print(
                f"Best Model Saved! "
                f"Acc: {best_acc:.2f}%"
            )


if __name__ == "__main__":
    main()