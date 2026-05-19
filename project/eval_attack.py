import torch

from train import (DEVICE, MODEL_DICT, CHECKPOINT_DIR, get_dataloaders)
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

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
    epsilon = 0.03

    attacks = {
        "FGSM": lambda model, x, y:
            fgsm_attack(model, x, y, epsilon=epsilon, device=DEVICE),

        "PGD": lambda model, x, y:
            pgd_attack(model, x, y, epsilon=epsilon, alpha=0.007, iters=10, device=DEVICE)
    }

    for model_name, model_class in MODEL_DICT.items():

        print(f"\nEvaluating {model_name}")
        model = model_class().to(DEVICE)
        checkpoint_path = (CHECKPOINT_DIR / f"{model_name}_best.pth")

        model.load_state_dict(
            torch.load(checkpoint_path, map_location=DEVICE)
        )

        for attack_name, attack_fn in attacks.items():

            acc = evaluate_attack(model, test_loader, attack_fn)
            print(
                f"{attack_name} Accuracy: "
                f"{acc:.2f}%"
            )

if __name__ == "__main__":
    main()