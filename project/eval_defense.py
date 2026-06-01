# Evaluate Defenses for Adversarially Trained Models

import argparse
import torch

from project.train import (DEVICE, MODEL_DICT, get_dataloaders, CHECKPOINT_DIR)
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

# Argument Parser
parser = argparse.ArgumentParser(description="Evaluate Defenses")
parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "cnn"])
parser.add_argument("--defense", type=str, default="trades", choices=["trades   ", "arow"])
args = parser.parse_args()

def evaluate_attack(model, loader, attack_fn):

    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        adv_images = attack_fn(model, images, labels)
        outputs = model(adv_images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100 * correct / total


def main():

    _, test_loader = get_dataloaders()
    model = MODEL_DICT[args.model]().to(DEVICE)

    checkpoint_path = (
        CHECKPOINT_DIR /
        f"{args.model}_{args.defense}.pth"
    )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)
    )

    attacks = {

        "FGSM": lambda model, x, y:
            fgsm_attack(model, x, y, epsilon=8/255, device=DEVICE),

        "PGD": lambda model, x, y:
            pgd_attack(model, x, y, epsilon=8/255, alpha=2/255, iters=10, device=DEVICE)
    }

    for attack_name, attack_fn in attacks.items():

        acc = evaluate_attack(model, test_loader, attack_fn)

        print(
            f"{attack_name} Robust Accuracy: "
            f"{acc:.2f}%"
        )


if __name__ == "__main__":
    main()