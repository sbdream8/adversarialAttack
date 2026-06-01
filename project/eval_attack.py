# Evaluate Adversarial Attacks on CIFAR10 Models

import argparse
import torch

from project.train import (DEVICE, MODEL_DICT, CHECKPOINT_DIR, get_dataloaders)
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


# Argument Parser
parser = argparse.ArgumentParser(description="Evaluate Robustness")
parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "cnn"])
parser.add_argument("--epsilon", type=float, default=8/255)

args = parser.parse_args()
print(args)


# Attack Evaluation
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


# Main
def main():

    _, test_loader = (get_dataloaders())

    model = MODEL_DICT[args.model]().to(DEVICE)

    checkpoint_path = (
        CHECKPOINT_DIR /
        f"{args.model}_best.pth"
    )

    model.load_state_dict(torch.load(checkpoint_path,map_location=DEVICE))

    attacks = {

        "FGSM": lambda model, x, y:
            fgsm_attack(model, x, y, epsilon=args.epsilon, device=DEVICE),

        "PGD": lambda model, x, y:
            pgd_attack(model, x, y, epsilon=args.epsilon, alpha=2/255, iters=10, device=DEVICE)
    }

    print(f"\nEvaluating {args.model}")

    for attack_name, attack_fn in attacks.items():

        acc = evaluate_attack(model, test_loader, attack_fn)
        print(
            f"{attack_name} Accuracy: "
            f"{acc:.2f}%"
        )


if __name__ == "__main__":
    main()